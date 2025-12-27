from httpcore import stream
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


llm=ChatGroq(model="openai/gpt-oss-120b")
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database= './db/chatbot_conversations.db', check_same_thread=False)

# Sqlite Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

## extracting number of threads from db

def retrive_all_threads():
    all_threads = set() ## for storing unique thread ids


    for checkpoint in checkpointer.list(None):
        # print(checkpointer)
        # print(checkpointer.config) 
        # # {'configurable': {'thread_id': 'thread-1', 'checkpoint_ns': '', 'checkpoint_id': '1f0e30d5-a6ce-60b6-bfff-33f608a826af'}}
        # print(checkpointer.config['configurable']['thread_id'])
        
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
    # print(list(all_threads))
        return list(all_threads)


## to save the thread name as well in the db
def save_thread_name(thread_id: str, name: str):
    """Save thread name to SQLite database."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_names (
            thread_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    cursor.execute(
        "INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)",
        (str(thread_id), name)
    )
    conn.commit()

def load_all_thread_names():
    """Load all thread names from SQLite database."""
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS thread_names (thread_id TEXT PRIMARY KEY, name TEXT NOT NULL)")
    cursor.execute("SELECT thread_id, name FROM thread_names")
    return dict(cursor.fetchall())

def get_thread_name(thread_id: str):
    """Get thread name from database, return 'New Chat' if not exists."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM thread_names WHERE thread_id = ?", (thread_id,))
    result = cursor.fetchone()
    return result[0] if result else "New Chat"



## test db creation and chat invocation

# CONFIG = {'configurable': {'thread_id': 'thread-1'}}


# response = chatbot.invoke(
#                     {'messages': [HumanMessage(content="What is my name?")]},
#                     config = CONFIG,
#                 )   

# print(response)