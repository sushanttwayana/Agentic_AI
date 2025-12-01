from langchain_mcp_adapters.client import MultiServerMCPClient
# Alternative: from langchain_mcp_adapters.multi_server import MCPMultiServerClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

import asyncio
import os

async def main():
    # Your servers config (ensure mathserver.py exists and weather server runs on :8000)
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["mathserver.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
            # "transport": "sse",
            # "host": "127.0.0.1",
            # "port": 8000,
        }
    })

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    
    
    tools = await client.get_tools()

    print((f"Available tools: {[tool.name for tool in tools]}"))
    model = ChatGroq(model="openai/gpt-oss-120b")
    agent = create_react_agent(model, tools)

    math_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 10 plus 20 multiplied by 3?"}]
    })

    weather_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What's the weather in Kathmandu?"}]
    })
    
    print("WEATHER RESPONSE:", weather_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
