import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage
from state import AgentState
from nodes import intent_router, support_node, lead_qualifier_node, agentic_node

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", intent_router)
workflow.add_node("support", support_node)
workflow.add_node("qualifier", lead_qualifier_node)
workflow.add_node("agent", agentic_node)

workflow.set_entry_point("router")

# Conditional Edges (Router logic)
def route_decision(state):
    # If the user asks a question, go to support, but stay in the loop (END of run, not END of app)
    if state["dialog_state"] == "support":
        return "support"
    return "qualifier"

workflow.add_conditional_edges(
    "router",
    route_decision,
    {"support": "support", "qualifier": "qualifier"}
)

# Conditional Edges (Qualifier logic)
def qualifier_decision(state):
    # Only go to agent if we are actually done collecting info
    if state["dialog_state"] == "completed":
        return "agent"
    return END # Otherwise, wait for more user input

workflow.add_conditional_edges(
    "qualifier",
    qualifier_decision,
    {"agent": "agent", END: END}
)

# Terminal Edges
workflow.add_edge("support", END)
workflow.add_edge("agent", END)

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

app = workflow.compile(checkpointer=checkpointer)

# --- 3. EXECUTION UTILITY ---
def run_chat(thread_id):
    print(f"\n--- Starting Chat (Thread: {thread_id}) ---")
    config = {"configurable": {"thread_id": thread_id}}
    
    current_state = app.get_state(config)
    if current_state.values:
        print("--- Resuming Conversation (Last 3 Messages) ---")
        messages = current_state.values.get("messages", [])
        for msg in messages[-3:]:
            if isinstance(msg, HumanMessage):
                print(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"Agent: {msg.content}")
        print("-----------------------------------------------")
        
        if current_state.values.get("dialog_state") == "finished":
            print("[System] This transaction was previously completed.")
            return
    else:
        print("--- New Conversation Started ---")

    print("Type 'quit' to exit.")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
            
        # Run the graph
        events = app.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config, 
            stream_mode="values"
        )
        
        # Track if we need to exit after this turn
        should_terminate = False
        
        for event in events:
            if "messages" in event:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    print(f"Agent: {last_msg.content}")

            if event.get("dialog_state") == "finished":
                should_terminate = True

        if should_terminate:
            print("\n--- Transaction Completed Successfully. Exiting Chat. ---")
            break

if __name__ == "__main__":
    run_chat(thread_id="user_session_3")