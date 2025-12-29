from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio    
import threading
import os

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="openai/gpt-oss-120b")

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=COLIC9J6JWQKFIMK"
    r = requests.get(url)
    return r.json()


client = MultiServerMCPClient(
    {
        "math_server": {
            "command": "C:/Users/Vanilla/anaconda3/Scripts/uv.exe",
            "args": [
                "run",
                "--directory",
                "G:/sushant/Model_Context_Protocol/chatbot_mcp",
                "fastmcp",
                "run",
                "math_server.py"
            ],
            "transport": "stdio"
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://expense-tracker-mcp-proj.fastmcp.app/mcp"
        },
    }
)
# client = MultiServerMCPClient(
#     {
#         "arith": {
#             "transport": "stdio",
#             "command": "python3",
#             "args": ["/Users/nitish/Desktop/mcp-math-server/main.py"],
#         },
#         "expense": {
#             "transport": "streamable_http",  # if this fails, try "sse"
#             "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
#         }
#     }
# )


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------
_checkpointer = None  # Global singleton

async def get_checkpointer():
    """Get or create the global checkpointer with proper connection lifecycle."""
    global _checkpointer
    if _checkpointer is None:
        import aiosqlite
        conn = await aiosqlite.connect("chatbot.db")
        await conn.__aenter__()  # Enter async context manually
        _checkpointer = AsyncSqliteSaver(conn)
    return _checkpointer

# Replace the old synchronous init
checkpointer = None  # Will be initialized on first use

# -------------------
# 6. Graph - FIXED
# -------------------
async def get_chatbot():
    """Get compiled chatbot with checkpointer."""
    global checkpointer
    if checkpointer is None:
        checkpointer = await get_checkpointer()
    
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    
    if tool_node:
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges("chat_node", tools_condition)
        graph.add_edge("tools", "chat_node")
    else:
        graph.add_edge("chat_node", END)
    
    return graph.compile(checkpointer=checkpointer)

# Export for frontend
chatbot = None  # Will be lazy-loaded

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    """List all thread IDs from checkpointer."""
    if checkpointer is None:
        cp = await get_checkpointer()
    else:
        cp = checkpointer
    all_threads = set()
    try:
        async for checkpoint in cp.alist(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
    except Exception:
        pass  # No threads yet is OK
    return list(all_threads)


def retrieve_all_threads():
    """Synchronous wrapper for frontend."""
    try:
        return run_async(_alist_threads())
    except:
        return []  # Graceful fallback for Streamlit init
