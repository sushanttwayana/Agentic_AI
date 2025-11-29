from mcp.server.fastmcp import FastMCP

mcp  = FastMCP("MathServer")

@mcp.tool()
def add(a:float, b:float) -> float:
    """Add two numbers"""

    return a + b

@mcp.tool()
def multiply(a:float, b:float) -> float:
    """Multiply two numbers"""

    return a*b



## The transport=="stdio" means that the server will run in the terminal and the client will connect to it using the standard input and output streams.
if __name__ == "__main__": 
    mcp.run(transport="stdio")


