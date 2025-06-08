#!/usr/bin/env python3
"""
Duck Detector MCP Server
========================

An MCP server that detects when the word "duck" is mentioned and provides
a prompt for the LLM to respond with "cuack".

Created following the MCP implementation patterns from the hackathon documentation.
"""

from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Duck Detector")


@mcp.tool()
def detect_duck_in_text(text: str) -> dict:
    """
    Detect if the word 'duck' appears in the provided text.
    
    Args:
        text: The text to analyze for duck mentions
        
    Returns:
        Dictionary with detection results and instructions
    """
    # Convert to lowercase for case-insensitive detection
    text_lower = text.lower()
    
    # Check if "duck" is mentioned
    duck_found = "duck" in text_lower
    
    if duck_found:
        return {
            "duck_detected": True,
            "message": "Duck detected! The user mentioned a duck.",
            "instruction": "You should now say 'cuack' in response to the duck mention.",
            "original_text": text
        }
    else:
        return {
            "duck_detected": False,
            "message": "No duck detected in the text.",
            "original_text": text
        }


@mcp.prompt()
def duck_response_prompt() -> str:
    """
    Provides a prompt template for responding when a duck is detected.
    
    Returns:
        A prompt instructing the LLM to say "cuack"
    """
    return """The user said Duck, now you need to say "cuack"

When a user mentions the word "duck" in any context, you should respond with "cuack" to acknowledge the duck reference. This is part of a fun interaction pattern in the Duck Detector MCP server."""


@mcp.prompt()
def duck_context_prompt(user_text: str) -> str:
    """
    Provides context-aware prompt when duck is detected in specific text.
    
    Args:
        user_text: The original text where duck was detected
        
    Returns:
        A formatted prompt with context
    """
    return f"""The user said: "{user_text}"

Since the word "duck" was detected in their message, you need to say "cuack" in response. This is a playful acknowledgment of the duck mention."""


# Run the server when executed directly
if __name__ == "__main__":
    # For direct execution - supports stdio transport for Claude Desktop
    mcp.run(transport="stdio")
