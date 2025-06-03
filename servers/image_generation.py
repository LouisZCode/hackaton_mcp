from typing import Any

#Gradio for the hackaton:
import gradio as gr

# we used   uv add mcp[cli] httpx   to get these:
import httpx
from mcp.server.fastmcp import FastMCP

#Actually needed, this defines the MCP and Decorators
mcp = FastMCP("image_get")

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


demo = gr.Interface(
    fn=tool_name,
    inputs=gr.Textbox(placeholder="this is a template"),
    outputs=gr.JSON(),
    title="Image Get MCP Server",
    description="This is just a Template"
)

if __name__ == "__main__":
    # Initialize and run the server
    demo.launch(mcp_server=True)