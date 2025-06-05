from typing import Any
import csv
import os

from langchain_community.retrievers import BM25Retriever

#Gradio for the hackaton:
import gradio as gr

# we used   uv add mcp[cli] httpx   to get these:
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("german-errors-db")

# Constants - get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "german_errors_db.csv")

#let's add our helper functions:

async def load_german_errors():
    """Load German errors database from CSV"""
    try:
        print(f"Looking for CSV at: {CSV_PATH}")  # Debug print
        with open(CSV_PATH, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return list(reader)
    except Exception as e:
        print(f"Error loading CSV: {e}")  # Debug print
        return [{"error": f"Could not load database: {str(e)}", "correct": "Check CSV path"}]

def search_german_errors(query: str = "") -> dict:
    """
    Search the German errors database
    
    Args:
        query (str): Search term (error type, word, etc.)
        
    Returns:
        dict: Search results from German errors database
    """
    import asyncio
    errors_db = asyncio.run(load_german_errors())
    
    if not query:
        # Return all entries if no query
        return {
            "total_entries": len(errors_db),
            "showing": "All entries",
            "results": errors_db
        }
    
    # Simple search in the database
    results = []
    for entry in errors_db:
        if query.lower() in str(entry).lower():
            results.append(entry)
    
    return {
        "query": query,
        "total_matches": len(results),
        "results": results  # Show all matching results
    }

@mcp.resource("uri://german-errors-db")
async def german_errors_resource():
    """The most typical German Errors and mistakes database
        Use this database to find common German errors and their corrections
    """
    return await load_german_errors()

demo = gr.Interface(
    fn=german_errors_resource,  # Function that returns the whole DB
    inputs=None,  # No input needed!
    outputs=gr.JSON(),
    title="German Errors Database MCP",
    description="Complete German errors database exposed via MCP protocol",
    submit_btn="expose resource!"
)

if __name__ == "__main__":
    # Initialize and run the server
    demo.launch(mcp_server=True)