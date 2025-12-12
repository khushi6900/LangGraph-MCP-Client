from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import asyncio
import json
from datetime import datetime
import uuid

load_dotenv()

# ============================================
# GLOBAL VARIABLES (Shared across everything)
# ============================================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

VECTOR_DB_PATH = "vector_store.faiss"
EMBEDDING_MODEL = "BAAI/bge-small-en"

# These MUST be at module level and accessed everywhere
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = None  # Will be loaded/created
ACTIVE_FILE_ID = None
chatbot = None
tool_dict = None

# Load existing vector DB
if os.path.exists(VECTOR_DB_PATH):
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    # print(f"Loaded existing vector DB with {db.index.ntotal} vectors")

# ============================================
# LLM SETUP
# ============================================
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

SYSTEM_PROMPT = SystemMessage(content=f"""
You are a helpful assistant with access to multiple tools.

Current date and time: {NOW}

You MUST use tools correctly based on these rules:

==========================
RULES FOR TOOL USAGE
==========================

1. CALCULATIONS  
   - If the user asks to add, subtract, multiply, divide, or use powers  
     → USE the appropriate calculator tool.

2. WEB SEARCH  
   - If the user asks for information that requires internet knowledge  
     → USE the tavily_search tool.

3. EXPENSE MANAGEMENT  
   - If the user asks to add expenses, list expenses, or summarize  
     → USE the expense tools.

4. RAG DOCUMENT QUERIES  
   - If the user asks a question that requires reading the **content** of the uploaded document  
     → USE the rag_query tool.

==========================
WHEN *NOT* TO USE RAG
==========================

Do NOT use rag_query when the user asks about:
- file name  
- file id  
- upload status  
- file metadata  
- whether a file exists  
- general questions about "the document" that are NOT content-based  

For these, answer directly using the metadata already stored in memory.  
If no file exists, politely say: "No document has been uploaded yet."

==========================
CRITICAL: EXPENSE TABLE FORMAT
==========================

When you use the list_expenses tool, you MUST respond with EXACTLY this format:

Here are your expenses:

| ID | Date | Category | Amount | Description |
|----|------|----------|--------|-------------|
| 1 | 2024-01-15 | Food | ₹15.00 | Lunch |
| 2 | 2024-01-16 | Transport | ₹8.50 | Bus fare |

MANDATORY RULES:
- Start with "Here are your expenses:" on its own line
- Then add a blank line
- Then the table MUST start with the header row: | ID | Date | Category | Amount | Description |
- Next line MUST be the separator: |----|------|----------|--------|-------------|
- Each expense row MUST follow: | ID | Date | Category | Amount | Description |
- Sort by ID in ASCENDING order (1, 2, 3, ...)
- Use the pipe character | to separate columns
- Format amounts as ₹XX.XX (e.g., ₹15.00, not 15 or 15.0)
- If no expenses, respond: "No expenses found."

DO NOT add any extra text after the table.
DO NOT use markdown code blocks (no ```).
DO NOT add bullet points or numbered lists.
ONLY the table format shown above.

==========================
AFTER TOOL EXECUTION
==========================

After calling a tool and receiving the result:
- For list_expenses: Use the EXACT table format above
- For other tools: Respond in clear, simple plain text
- Use only normal characters: + - * / = numbers and words
- NO LaTeX, no emojis

==========================
RESPONSE FORMAT RULES
==========================

1. For normal questions, explanations, definitions, summaries, or general answers:
   - Respond in normal paragraph text
   - Do NOT create tables or markdown-like formatting

2. Only produce a table when list_expenses tool is used
   - Must follow the EXACT format in CRITICAL section above

==========================
EXAMPLES
==========================

User: "search latest AI news"  
Assistant: (call tavily_search tool)  
Then summarize the results in plain text.

User: "add expense for lunch 15 dollars"  
Assistant: (call add_expense tool)  
Then confirm: "Added expense: lunch - ₹15.00"

User: "list all expenses" or "show my expenses"
Assistant: (call list_expenses tool)
Then respond EXACTLY like this:

Here are your expenses:

| ID | Date | Category | Amount | Description |
|----|------|----------|--------|-------------|
| 1 | 2024-01-15 | Food | ₹15.00 | Lunch |
| 2 | 2024-01-16 | Transport | ₹8.50 | Bus fare |

User: "What does my uploaded document say about neural networks?"  
Assistant: (call rag_query tool with a question)

User: "What is the name of the uploaded file?"  
Assistant: DO NOT call rag_query.  
Assistant: Answer directly from metadata.

==========================

Follow these rules strictly.
When listing expenses, you MUST use the table format shown above.
Be concise, accurate, and always choose the correct tool.
""")
# ============================================
# MCP SERVER CONFIG
# ============================================
SERVERS = {
    "Calculator_Server": {
        "transport": "stdio",
        "command": "uv",
        "args": ["run", "fastmcp", "run", "D:/Users/Khushi.Mahajan/Documents/Projects/MCP_Client_Server/Calculator_server/main.py"]
    },
    "Expense_Server": {
        "transport": "stdio",
        "command": "uv",
        "args": ["run", "fastmcp", "run", "D:/Users/Khushi.Mahajan/Documents/Projects/test_remote_mcp_server/local_expense_tracking_server.py"]
    }
}

client = MultiServerMCPClient(SERVERS)


# ============================================
# RAG FUNCTIONS
# ============================================

def extract_text(file_path: str) -> str:
    """Extract text from PDF, TXT, DOCX."""
    # print(f"\nExtracting text from: {file_path}")
    print(f"   File exists: {os.path.exists(file_path)}")
    
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            # print(f" TXT extracted: {len(text)} chars")
            return text

        elif file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            # print(f" PDF extracted: {len(text)} chars from {len(reader.pages)} pages")
            return text
        
        elif file_path.endswith(".docx"):
            from docx import Document
            doc = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)
            # print(f" DOCX extracted: {len(text)} chars")
            return text
        
        else:
            # print(f" Unsupported file type: {file_path}")
            return ""
            
    except Exception as e:
        # print(f"Error extracting text: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

def process_uploaded_file(file_path: str, file_id: str):
    """Extract → Split → Embed → Save to FAISS."""
    global db, ACTIVE_FILE_ID
    
    print(f"\n PROCESSING: {file_path}")
    
    raw_text = extract_text(file_path)
    if not raw_text:
        return
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(raw_text)
    metadatas = [{"file_id": file_id} for _ in chunks]
    
    # print(f"Creating {len(chunks)} embeddings...")
    
    # CLEAR OLD DATABASE (for testing - remove this later)
    import shutil
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH, ignore_errors=True)
    
    # Create fresh database
    db = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)
    db.save_local(VECTOR_DB_PATH)
    
    # Reload
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Verify
    test = db.similarity_search("test", k=5)
    matching = [d for d in test if d.metadata.get("file_id") == file_id]
    print(f" Verified: {len(matching)} docs with file_id")
    
    ACTIVE_FILE_ID = file_id


@tool
def rag_query(query: str) -> str:
    """Retrieve answer from the uploaded document. Use this to answer questions about document content."""
    global db, ACTIVE_FILE_ID
    
    # print(f"\n{'='*60}")
    # print(f" RAG_QUERY CALLED")
    # print(f"Query: {query}")
    # print(f"ACTIVE_FILE_ID: {ACTIVE_FILE_ID}")
    # print(f"db exists: {db is not None}")
    if db:
        print(f"Total vectors in db: {db.index.ntotal}")
    print(f"{'='*60}\n")

    if db is None:
        return "ERROR: No knowledge base found. The document upload may have failed."

    if not ACTIVE_FILE_ID:
        return "ERROR: No active document ID set. The upload process may have failed."

    # Search WITHOUT filter first
    # print(f" Searching for: {query}")
    all_docs = db.similarity_search(query, k=10)
    # print(f" Found {len(all_docs)} documents total")
    
    for i, doc in enumerate(all_docs[:3]):
        print(f"  Doc {i}: file_id={doc.metadata.get('file_id')}, content={doc.page_content[:100]}...")
    
    # Filter by active file
    filtered_docs = [d for d in all_docs if d.metadata.get("file_id") == ACTIVE_FILE_ID]
    # print(f" After filtering for file_id {ACTIVE_FILE_ID}: {len(filtered_docs)} documents")
    
    if not filtered_docs:
        return f"No relevant information found in the document for: {query}"

    context = "\n\n".join([d.page_content for d in filtered_docs[:5]])
    # print(f" Returning {len(context)} characters of context") 
    # print(f"{'='*60}\n")
    return context[:5000]


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def setup_graph():
    global tool_dict
    
    tools = await client.get_tools()
    tavily_tool = TavilySearch(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_results=2
    )
    tavily_tool.description = "Web search tool. Use ONLY: {\"query\": \"...\"}."
    
    tools.append(tavily_tool)
    tools.append(rag_query)
    
    tool_dict = {tool.name: tool for tool in tools}
    # print(f" Available Tools: {list(tool_dict.keys())}")

    llm_with_tools = model.bind_tools(tools)
    
    async def chat_node(state: ChatState):
        # print(f"\n CHAT NODE: {len(state['messages'])} messages")
        # print(f"   Last: {state['messages'][-1].content[:100] if state['messages'] else 'None'}...")
        # print(f"   Global ACTIVE_FILE_ID: {ACTIVE_FILE_ID}\n")
        
        messages = [SYSTEM_PROMPT] + state["messages"]
        
        accumulated_content = ""
        accumulated_tool_calls = []
        
        async for chunk in llm_with_tools.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                accumulated_content += chunk.content
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                accumulated_tool_calls.extend(chunk.tool_calls)
        
        if accumulated_tool_calls:
            complete_message = AIMessage(
                content=accumulated_content,
                tool_calls=accumulated_tool_calls
            )
        else:
            complete_message = AIMessage(content=accumulated_content)
            # print(f"MODEL RESPONSE: {accumulated_content[:100]}...\n")
        
        return {"messages": [complete_message]}

    async def custom_tool_node(state: ChatState):
        last_msg = state["messages"][-1]

        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return {"messages": []}

        tool_call = last_msg.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tools = await client.get_tools()
        tools.append(tavily_tool)
        tools.append(rag_query)
        
        # Create tool dictionary
        tool_dict = {tool.name: tool for tool in tools}
        # print("Available Tools:", list(tool_dict.keys()))

        # print(f"\n{'='*60}")
        # print(f"EXECUTING TOOL: {tool_name}")
        # print(f"   Args: {tool_args}")
        # print(f"   Global ACTIVE_FILE_ID: {ACTIVE_FILE_ID}")
        # print(f"{'='*60}\n")

        tool = tool_dict.get(tool_name)
        if not tool:
            result = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                if hasattr(tool, "ainvoke"):
                    result = await tool.ainvoke(tool_args)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: tool.invoke(tool_args)
                    )
                # print(f"Tool result: {str(result)[:200]}...\n")
            except Exception as e:
                result = f"Error: {str(e)}"
                import traceback
                traceback.print_exc()

        return {
            "messages": [
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
            ]
        }
    
    def should_use_tools(state: ChatState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    conn = await aiosqlite.connect("chatbot.db")
    checkpointer = AsyncSqliteSaver(conn=conn)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", custom_tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", should_use_tools, {"tools": "tools", "__end__": "__end__"})
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile(checkpointer=checkpointer)

    # RETURN BOTH VALUES
    return chatbot, tool_dict

# -------------------
# 9. Streaming for Frontend
# -------------------

async def stream_graph_updates(chatbot, initial_state, config):
    """
    Stream updates to frontend while preserving tool calling functionality.
    This streams state updates, not individual tokens.
    """
    async for event in chatbot.astream(initial_state, config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if "messages" in node_output:
                for message in node_output["messages"]:
                    # Check if it's a streaming token or complete message
                    if isinstance(message, AIMessage):
                        if message.content.startswith("TOOL_EVENT::"):
                            # Tool is being called
                            yield {"type": "tool_call", "content": message.content}
                        elif hasattr(message, "tool_calls") and message.tool_calls:
                            # Tool call message
                            yield {"type": "tool_request", "tool_calls": message.tool_calls}
                        else:
                            # Regular AI response
                            yield {"type": "ai_message", "content": message.content}
                    elif isinstance(message, ToolMessage):
                        # Tool result
                        yield {"type": "tool_result", "content": message.content, "tool": message.name}




async def load_chat_history(checkpointer, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = await checkpointer.aget(config)
    if checkpoint:
        return checkpoint.get("channel_values", {}).get("messages", [])
    return []

# async def main():
#     chatbot, tool_dict = await setup_graph()

#     # Example interaction
#     user_message = "add the expense of 80rs for food today"
#     initial_state: ChatState = {"messages": [HumanMessage(content=user_message)]}
#     config = {"configurable": {"thread_id": "chat-6"}}

#     print(f"User: {user_message}\n")
    
#     # Stream the conversation
#     print("=== STREAMING UPDATES ===")
#     async for update in stream_graph_updates(chatbot, initial_state, config):
#         print(f"Update: {update}")
#     print("=========================\n")
    
#     # Get final state
#     final_state = await chatbot.aget_state(config)
    
#     print("\n=======================================================================")
#     print("=== FINAL STATE ===")
#     for i, msg in enumerate(final_state.values["messages"]):
#         print(f"\nMessage {i}: {type(msg).__name__}")
#         print(f"Content: {msg.content[:100] if len(msg.content) > 100 else msg.content}")
#         if hasattr(msg, "tool_calls") and msg.tool_calls:
#             print(f"Tool calls: {msg.tool_calls}")
#     print("=======================================================================\n")
    
#     if final_state.values["messages"]:
#         last_message = final_state.values["messages"][-1]
#         print(f"Chatbot Final Response: {last_message.content}")
    
#     await chatbot.checkpointer.conn.close()

# if __name__ == "__main__":
#     asyncio.run(main())