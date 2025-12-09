from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_tavily import TavilySearch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os
import asyncio
import json

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

SYSTEM_PROMPT = SystemMessage(content="""You are a helpful assistant with access to tools for calculations, expense tracking, and web search.

When users ask you to:
1. Perform calculations (add, subtract, multiply, divide, power) - USE THE APPROPRIATE CALCULATOR TOOL
2. Search for information - USE THE TAVILY_SEARCH TOOL  
3. Manage expenses (add expense, list expenses, summarize) - USE THE EXPENSE TOOLS

IMPORTANT INSTRUCTIONS:
- When you need to use a tool, you MUST call the tool with the correct arguments
- After getting tool results, present the answer in simple plain text
- Use only normal characters: +, -, *, /, =, numbers, and words
- NO LaTeX, markdown, or special symbols
- Keep responses concise and helpful

TOOL USAGE EXAMPLES:
User: "add 34 and 567 using the tool"
You: [use add tool with a=34, b=567] → Then say: "34 + 567 = 601"

User: "search for latest AI news"
You: [use tavily_search tool] → Then summarize results in plain text

User: "add expense for lunch $15"
You: [use add_expense tool] → Then confirm: "Added expense: lunch - $15"

Remember: You have tools available. Use them when appropriate!
""")

# -------------------
# 2. MCP Servers
# -------------------

SERVERS = {
    "Calculator_Server": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            "D:/Users/Khushi.Mahajan/Documents/Projects/MCP_Client_Server/Calculator_server/main.py"
        ]
    },
    "Expense_Server": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            "D:/Users/Khushi.Mahajan/Documents/Projects/test_remote_mcp_server/local_expense_tracking_server.py"
        ]
    },
}

# -------------------
# 3. MCP Client
# -------------------

client = MultiServerMCPClient(SERVERS)

# -------------------
# 4. Tools
# -------------------
tavily_tool = TavilySearch(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    max_results=2
)

# Prevent LLM from overriding parameters
tavily_tool.description = """
Web search tool. Use ONLY: {"query": "..."}.
Do NOT include topic or search_depth.
"""
    
# -------------------
# 5. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def setup_graph():
    tools = await client.get_tools()
    tools.append(tavily_tool)
    
    # Create a dictionary of tools for easy access
    tool_dict = {tool.name: tool for tool in tools}
    print("Available Tools:", list(tool_dict.keys()))

    llm_with_tools = model.bind_tools(tools)
    
    # -------------------
    # 6. Nodes
    # -------------------
    async def chat_node(state: ChatState):
        """LLM node that streams tokens AND preserves tool calls."""
        messages = [SYSTEM_PROMPT] + state["messages"]
        
        # Accumulate the complete response
        accumulated_content = ""
        accumulated_tool_calls = []
        final_response = None
        
        # Stream the response
        async for chunk in llm_with_tools.astream(messages):
            # Accumulate content
            if hasattr(chunk, "content") and chunk.content:
                accumulated_content += chunk.content
            
            # Accumulate tool calls (they come in chunks too)
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                accumulated_tool_calls.extend(chunk.tool_calls)
            
            # Keep the last chunk as it has the most complete data
            final_response = chunk
        
        # Create the final complete message
        if accumulated_tool_calls:
            # Model wants to call a tool
            complete_message = AIMessage(
                content=accumulated_content,
                tool_calls=accumulated_tool_calls
            )
            print(f"\n=== MODEL WANTS TO CALL TOOL ===")
            print(f"Tool calls: {accumulated_tool_calls}")
            print(f"=================================\n")
        else:
            # Regular text response
            complete_message = AIMessage(content=accumulated_content)
            print(f"\n=== MODEL TEXT RESPONSE ===")
            print(f"Content: {accumulated_content[:100]}...")
            print(f"===========================\n")
        
        # Return the complete message (this is what gets checked by tools_condition)
        return {"messages": [complete_message]}

    async def custom_tool_node(state: ChatState):
        last_msg = state["messages"][-1]

        # If no tool call → nothing to do
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return {"messages": []}

        tool_call = last_msg.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # PRINT TOOL EXECUTED
        print(f"\n=== TOOL EXECUTION ===")
        print(f"Tool: {tool_name}")
        print(f"Arguments: {tool_args}")
        print(f"======================\n")

        # Create a new "ToolEvent" message for frontend
        tool_event = AIMessage(
            content=f"TOOL_EVENT::{tool_name}::{json.dumps(tool_args)}"
        )
        
        # Execute tool
        tool = tool_dict.get(tool_name)
        if not tool:
            result = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                if hasattr(tool, "ainvoke"):
                    result = await tool.ainvoke(tool_args)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(None, lambda: tool.run(tool_args))
                
                print(f"Tool result: {result}\n")
            except Exception as e:
                result = f"Error executing tool '{tool_name}': {str(e)}"
                print(f"Tool error: {result}\n")

        # Return tool response
        return {
            "messages": [
                tool_event,
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
            ]
        }

    # -------------------
    # 7. Checkpointer
    # -------------------
    conn = await aiosqlite.connect("chatbot.db")
    checkpointer = AsyncSqliteSaver(conn=conn)

    # -------------------
    # 8. Graph
    # -------------------
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", custom_tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools", "__end__": "__end__"})
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile(checkpointer=checkpointer)

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

# -------------------
# 10. Helper Functions
# -------------------

async def load_chat_history(checkpointer, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = await checkpointer.aget(config)
    if checkpoint:
        return checkpoint.get("channel_values", {}).get("messages", [])
    return []

async def main():
    chatbot, tool_dict = await setup_graph()

    # Example interaction
    user_message = "add the expense of 40rs for icecream today"
    initial_state: ChatState = {"messages": [HumanMessage(content=user_message)]}
    config = {"configurable": {"thread_id": "chat-6"}}

    print(f"User: {user_message}\n")
    
    # Stream the conversation
    print("=== STREAMING UPDATES ===")
    async for update in stream_graph_updates(chatbot, initial_state, config):
        print(f"Update: {update}")
    print("=========================\n")
    
    # Get final state
    final_state = await chatbot.aget_state(config)
    
    print("\n=======================================================================")
    print("=== FINAL STATE ===")
    for i, msg in enumerate(final_state.values["messages"]):
        print(f"\nMessage {i}: {type(msg).__name__}")
        print(f"Content: {msg.content[:100] if len(msg.content) > 100 else msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"Tool calls: {msg.tool_calls}")
    print("=======================================================================\n")
    
    if final_state.values["messages"]:
        last_message = final_state.values["messages"][-1]
        print(f"Chatbot Final Response: {last_message.content}")
    
    await chatbot.checkpointer.conn.close()

if __name__ == "__main__":
    asyncio.run(main())