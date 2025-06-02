from typing import Any
# we used   uv add mcp[cli] httpx   to get these:
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("name")  #this param its only the name of the app

# Constants


#let’s add our helper functions:

async def helper_function():
    """description"""

    #return 
    pass



@mcp.tool()
async def tool_name(state: str) -> str:
    """tool_description

    Args:
        arg1: arg1_description
        arg2: arg2_description
    """
    pass


@mcp.resource("config://version")
async def resource_name():
    """resource_description
    """
    pass


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') #define the needs