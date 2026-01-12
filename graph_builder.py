# graph_builder.py
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state import AgentState
from nodes import intent_router, support_node, lead_qualifier_node, agentic_node

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("router", intent_router)
    workflow.add_node("support", support_node)
    workflow.add_node("qualifier", lead_qualifier_node)
    workflow.add_node("agent", agentic_node)

    workflow.set_entry_point("router")

    # Conditional Edges
    def route_decision(state):
        if state["dialog_state"] == "support":
            return "support"
        return "qualifier"

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {"support": "support", "qualifier": "qualifier"}
    )

    def qualifier_decision(state):
        if state["dialog_state"] == "completed":
            return "agent"
        return END

    workflow.add_conditional_edges(
        "qualifier",
        qualifier_decision,
        {"agent": "agent", END: END}
    )

    workflow.add_edge("support", END)
    workflow.add_edge("agent", END)

    return workflow.compile(checkpointer=checkpointer)

# Create a global instance
graph_app = build_graph()