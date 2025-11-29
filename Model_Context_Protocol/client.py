import asyncio
from langchain_mcp import MCPToolkit
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

async def main():

    toolkit = MCPToolkit.toollist_from_servers({
        "math": {
            "command": "python",
            "args": ["mathserver.py"],
            "transport": "stdio"
        },
        "weather": {
            "url": "http://localhost:8000/weather",
            "transport": "streamable-http"
        }
    })

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    agent = create_react_agent(model, toolkit)

    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 10 + 20 * 3?"}]
    })

    print("RESPONSE:", response["messages"][-1].content)

asyncio.run(main())
