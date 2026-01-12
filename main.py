# main.py
from langchain_core.messages import HumanMessage, AIMessage
from tools.graph_builder import graph_app  # <--- Changed import

# --- EXECUTION UTILITY ---
def run_chat(thread_id):
    print(f"\n--- Starting Chat (Thread: {thread_id}) ---")
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check history using the imported app
    current_state = graph_app.get_state(config)
    
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
            
        # Run the graph using the imported app
        events = graph_app.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config, 
            stream_mode="values"
        )
        
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