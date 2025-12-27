from httpcore import stream
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

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

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


##  ==================================    streaming in langgraph
# stream = chatbot.stream(
#     {'messages': [HumanMessage(content="What is capital of Nepal? ")]},
#     stream_mode = 'messages'
# )
# for message_chunk,metadata in chatbot.stream(
    # {'messages': [HumanMessage(content="What is capital of Nepal? ")]},
    # config = {'configurable': {'thread_id': 'thread-1'}},
    # stream_mode = 'messages'
# ):


#     if message_chunk.content:
#         print(message_chunk.content, end='', flush=True)
        
# print(stream)
# print(type(stream))


### for resume chat feature ==============================================
# CONFIG = {'configurable': {'thread_id': 'thread-1'}}


# response = chatbot.invoke(
#                     {'messages': [HumanMessage(content="Hi my name is sushant")]},
#                     config = CONFIG,
#                 )     

# print(chatbot.get_state(config=CONFIG))
# print(chatbot.get_state(config=CONFIG).values['messages'])