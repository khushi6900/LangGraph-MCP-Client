from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from Chatbot_with_mcp import setup_graph, load_chat_history, process_uploaded_file
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import asyncio
import json
import logging
import os
import uuid

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Global ------------------
chatbot = None
tool_dict = None
db = None
ACTIVE_FILE_ID = None

class ChatRequest(BaseModel):
    thread_id: str
    message: str


# ------------------ Startup ------------------
@app.on_event("startup")
async def startup_event():
    global chatbot, tool_dict
    chatbot, tool_dict = await setup_graph()
    logger.info("Chatbot initialized!")
    logger.info(f"Available tools: {list(tool_dict.keys())}")


# ------------------ Routes ------------------
@app.get("/")
def root():
    return {"message": "Chatbot API running"}

# ===================================================================
#                   File Upload ENDPOINT
# ===================================================================

# Store active file_id (in memory; later you can use session storage)
@app.post("/upload/{thread_id}")
async def upload_file(thread_id: str, file: UploadFile = File(...)):
    global ACTIVE_FILE_ID, db  
    
    print(f"UPLOAD REQUEST")
    print(f"   Thread ID: {thread_id}")
    print(f"   Filename: {file.filename}")
    print(f"   Content-Type: {file.content_type}")
    
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        file_content = await file.read()
        print(f"Received {len(file_content)} bytes")
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        print(f"Saved to: {file_path}")
        print(f"File size on disk: {os.path.getsize(file_path)} bytes")
        
        # Process the file
        process_uploaded_file(file_path, file_id)
        

        if db:
            print(f"Total vectors: {db.index.ntotal}")
            
            # Test search
            test_results = db.similarity_search("test", k=3)
            print(f"   Test search returned: {len(test_results)} results")
            for i, doc in enumerate(test_results):
                print(f"     Result {i}: file_id={doc.metadata.get('file_id')}")
        
        return {
            "message": "File uploaded & processed",
            "file_id": file_id,
            "filename": file.filename,
            "active_file_id": ACTIVE_FILE_ID,
            "total_vectors": db.index.ntotal if db else 0,
            "file_size": os.path.getsize(file_path)
        }
        
    except Exception as e:
        print(f"\nERROR in upload endpoint:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "message": "Upload failed",
            "active_file_id": ACTIVE_FILE_ID,
            "total_vectors": db.index.ntotal if db else 0
        }

# ===================================================================
#                   MAIN STREAMING ENDPOINT
# ===================================================================

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streams the chatbot response with proper tool call support.
    Returns Server-Sent Events (SSE) with the following event types:
    - token: Individual text tokens from AI response
    - tool_call: When a tool is being called
    - tool_result: Result from tool execution
    - done: End of stream
    """
    async def event_generator():
        try:
            logger.info(f"Starting stream for thread: {req.thread_id}, message: {req.message}")

            new_message = HumanMessage(content=req.message)
            config = {"configurable": {"thread_id": req.thread_id}}
            
            # Track what we've seen
            current_content = ""
            has_streamed_content = False
            
            # Use astream with stream_mode="updates" to get state updates
            async for event in chatbot.astream(
                {"messages": [new_message]},
                config=config,
                stream_mode="updates"
            ):
                logger.debug(f"Received event: {event.keys()}")
                
                for node_name, node_output in event.items():
                    logger.debug(f"Processing node: {node_name}")
                    
                    if "messages" not in node_output:
                        continue
                    
                    for message in node_output["messages"]:
                        # Handle AI Messages
                        if isinstance(message, AIMessage):
                            # Check for tool event marker
                            if isinstance(message.content, str) and message.content.startswith("TOOL_EVENT::"):
                                try:
                                    parts = message.content.split("::", 2)
                                    if len(parts) == 3:
                                        _, tool_name, raw_args = parts
                                        args = json.loads(raw_args)
                                        
                                        logger.info(f"Tool called: {tool_name} with args: {args}")
                                        
                                        payload = {
                                            "type": "tool_call",
                                            "tool": tool_name,
                                            "args": args
                                        }
                                        yield f"data: {json.dumps(payload)}\n\n"
                                except Exception as e:
                                    logger.error(f"Error parsing tool event: {e}")
                            
                            # Check for tool_calls attribute (direct tool call)
                            elif hasattr(message, "tool_calls") and message.tool_calls:
                                for tool_call in message.tool_calls:
                                    logger.info(f"Tool requested: {tool_call}")
                                    payload = {
                                        "type": "tool_call",
                                        "tool": tool_call.get("name", "unknown"),
                                        "args": tool_call.get("args", {})
                                    }
                                    yield f"data: {json.dumps(payload)}\n\n"
                            
                            # Regular AI text response
                            elif message.content and not message.content.startswith("TOOL_EVENT::"):
                                # Only stream new content
                                if message.content != current_content:
                                    new_text = message.content[len(current_content):]
                                    current_content = message.content
                                    
                                    # Stream character by character
                                    for char in new_text:
                                        has_streamed_content = True
                                        payload = {"type": "token", "token": char}
                                        yield f"data: {json.dumps(payload)}\n\n"
                                        # Small delay for smoother streaming
                                        await asyncio.sleep(0.01)
                        
                        # Handle Tool Results
                        elif isinstance(message, ToolMessage):
                            logger.info(f"Tool result from {message.name}: {message.content}")
                            payload = {
                                "type": "tool_result",
                                "tool": message.name,
                                "result": message.content
                            }
                            yield f"data: {json.dumps(payload)}\n\n"
            
            # Send done signal
            logger.info("Stream completed successfully")
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            error_payload = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*"
        }
    )


# ===================================================================
#                    THREAD MANAGEMENT
# ===================================================================

@app.get("/threads")
async def get_threads():
    """Get all available chat threads with summaries."""
    if chatbot is None:
        raise HTTPException(500, "Chatbot not initialized")

    try:
        rows = await chatbot.checkpointer.conn.execute_fetchall("""
            SELECT DISTINCT thread_id
            FROM checkpoints
        """)

        thread_ids = [row[0] for row in rows if row[0] is not None]
        threads_with_summaries = []

        for thread_id in thread_ids:
            messages = await load_chat_history(chatbot.checkpointer, thread_id)

            summary = "New Chat"
            for msg in messages:
                if isinstance(msg, HumanMessage) and msg.content:
                    summary = msg.content[:50] + '...' if len(msg.content) > 50 else msg.content
                    break

            threads_with_summaries.append({
                "thread_id": thread_id,
                "summary": summary
            })

        return {"threads": threads_with_summaries}
    
    except Exception as e:
        logger.error(f"Error fetching threads: {e}", exc_info=True)
        raise HTTPException(500, f"Error fetching threads: {str(e)}")


@app.get("/thread/{thread_id}")
async def get_thread(thread_id: str):
    """Get all messages for a specific thread."""
    if chatbot is None:
        raise HTTPException(500, "Chatbot not initialized")

    try:
        messages = await load_chat_history(chatbot.checkpointer, thread_id)
        output = []

        for msg in messages:
            # Skip ToolMessage in history display
            if isinstance(msg, ToolMessage):
                continue

            # Handle tool event markers
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.startswith("TOOL_EVENT::"):
                try:
                    _, tool_name, raw_args = msg.content.split("::", 2)
                    output.append({
                        "type": "ToolEvent",
                        "tool": tool_name,
                        "args": json.loads(raw_args)
                    })
                except Exception as e:
                    logger.error(f"Error parsing tool event in history: {e}")
                    output.append({
                        "type": "ToolEvent",
                        "tool": "unknown",
                        "args": {}
                    })
                continue

            # Regular messages
            output.append({
                "type": msg.__class__.__name__,
                "role": getattr(msg, "role", None),
                "content": msg.content
            })

        return {"messages": output}
    
    except Exception as e:
        logger.error(f"Error fetching thread: {e}", exc_info=True)
        raise HTTPException(500, f"Error fetching thread: {str(e)}")


@app.delete("/thread/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a specific thread."""
    if chatbot is None:
        raise HTTPException(500, "Chatbot not initialized")

    try:
        await chatbot.checkpointer.conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        )
        await chatbot.checkpointer.conn.commit()
        logger.info(f"Deleted thread: {thread_id}")
        return {"deleted": True, "thread_id": thread_id}
    
    except Exception as e:
        logger.error(f"Error deleting thread: {e}", exc_info=True)
        raise HTTPException(500, f"Error deleting thread: {str(e)}")


# ===================================================================
#                   DEBUG ENDPOINTS
# ===================================================================

@app.post("/chat/debug")
async def chat_debug(request: ChatRequest):
    """Debug endpoint to see raw chatbot state."""
    if chatbot is None:
        raise HTTPException(500, "Chatbot not initialized")
    
    try:
        # Get current history
        history_before = await load_chat_history(chatbot.checkpointer, request.thread_id)
        
        # Run chatbot
        final_state = await chatbot.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.thread_id}},
        )
        
        # Return raw state
        messages = []
        for msg in final_state.get("messages", []):
            msg_data = {
                "type": type(msg).__name__,
                "content": str(msg.content) if hasattr(msg, 'content') else str(msg),
            }
            
            # Add tool_calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_data["tool_calls"] = msg.tool_calls
            
            messages.append(msg_data)
        
        return {
            "history_before_count": len(history_before),
            "total_messages": len(final_state.get("messages", [])),
            "messages": messages
        }
    
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}", exc_info=True)
        raise HTTPException(500, f"Debug error: {str(e)}")


@app.get("/tools")
async def get_tools():
    """Get list of available tools."""
    if tool_dict is None:
        raise HTTPException(500, "Tools not initialized")
    
    tools_info = []
    for tool_name, tool in tool_dict.items():
        tools_info.append({
            "name": tool_name,
            "description": getattr(tool, "description", "No description available")
        })
    
    return {"tools": tools_info}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot is not None,
        "tools_available": len(tool_dict) if tool_dict else 0
    }