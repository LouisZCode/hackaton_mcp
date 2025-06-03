from typing import Any

from langchain_community.retrievers import BM25Retriever

#Gradio for the hackaton:
import gradio as gr

# we used   uv add mcp[cli] httpx   to get these:
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
#Not being used as we are using Gradio for the Hackaton
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


demo = gr.Interface(
    fn=tool_name, #this is the function being used
    inputs=gr.Textbox(placeholder="this is a template"),   #should match the number of Arguments
    outputs=gr.JSON(),   #should match the number of return values
    title="Template for the MCPs in Gradio",
    description="This is just a Template"
)

if __name__ == "__main__":
    # Initialize and run the server
    demo.launch(mcp_server=True)