import uuid
import sqlite3
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from graph_builder import graph_app

app = FastAPI(title="Insurance Agent API")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    is_finished: bool
    dialog_state: str

class MessageHistory(BaseModel):
    role: str
    content: str

# --- DB Helper for listing threads ---
def get_all_thread_ids():
    """Query the SQLite checkpoint DB for all unique thread IDs."""
    try:
        # We connect specifically to read the threads
        conn = sqlite3.connect("checkpoints.db")
        cursor = conn.cursor()
        # The table name created by SqliteSaver is usually 'checkpoints'
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        return threads
    except Exception as e:
        print(f"Error fetching threads: {e}")
        return []

# --- Helper Logic ---
async def process_chat(thread_id: str, user_input: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    current_state = graph_app.get_state(config)
    if current_state.values and current_state.values.get("dialog_state") == "finished":
         return ChatResponse(
            response="Transaction previously completed. Please start a new chat.",
            thread_id=thread_id,
            is_finished=True,
            dialog_state="finished"
        )

    final_response_text = ""
    current_dialog_state = "unknown"
    
    events = graph_app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values"
    )

    for event in events:
        if "messages" in event:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage):
                final_response_text = last_msg.content
        if "dialog_state" in event:
            current_dialog_state = event["dialog_state"]

    is_finished = (current_dialog_state == "finished")

    return ChatResponse(
        response=final_response_text,
        thread_id=thread_id,
        is_finished=is_finished,
        dialog_state=current_dialog_state
    )

# --- API Endpoints ---

@app.get("/api/threads")
async def list_threads():
    return {"threads": get_all_thread_ids()}

@app.get("/api/chat/{thread_id}/history")
async def get_chat_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    current_state = graph_app.get_state(config)
    
    history = []
    if current_state.values:
        messages = current_state.values.get("messages", [])
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "agent", "content": msg.content})
    return {"history": history}

@app.post("/api/chat/new", response_model=ChatResponse)
async def start_new_chat(request: ChatRequest):
    thread_id = str(uuid.uuid4())
    return await process_chat(thread_id, request.message)

@app.post("/api/chat/{thread_id}", response_model=ChatResponse)
async def continue_chat(thread_id: str, request: ChatRequest):
    return await process_chat(thread_id, request.message)

# --- Serve Frontend ---
# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)