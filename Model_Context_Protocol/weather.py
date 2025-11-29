from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")


@mcp.tool()
async def get_weather(location:str)-> str:
    """Get the weather fo the location"""

    return f"Its sunny in Kathmandu"

## The transport="streamable-http" means that the server will run in the terminal and the client will connect to it using the standard input and output streams.
if __name__ == "__main__":
    mcp.run(transport="streamable-http")