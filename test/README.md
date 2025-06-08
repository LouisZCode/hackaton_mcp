# Duck Detector MCP Server 🦆

An MCP server that detects when the word "duck" is mentioned and provides prompts for the LLM to respond with "cuack".

## Features

- **Duck Detection Tool**: Analyzes text for "duck" mentions (case-insensitive)
- **Response Prompts**: Provides instructions for the LLM to say "cuack"
- **Context Awareness**: Includes the original text in the prompt context

## MCP Components

### Tools
- `detect_duck_in_text(text: str)` - Detects if "duck" appears in the provided text

### Prompts
- `duck_response_prompt()` - General instruction to say "cuack" when duck is detected
- `duck_context_prompt(user_text: str)` - Context-aware prompt with original text

## Usage

### Testing Locally
```bash
# Run the test script to see the duck detection in action
python test_duck_detector.py
```

### Running with MCP Inspector
```bash
# From the hackaton_mcp directory
mcp dev test/duck_detector_mcp.py
```

### Installing in Claude Desktop
```bash
# From the hackaton_mcp directory
mcp install test/duck_detector_mcp.py --name "Duck Detector"
```

### Manual Claude Desktop Configuration
Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "duck-detector": {
      "command": "/path/to/your/.venv/Scripts/python.exe",
      "args": ["/path/to/hackaton_mcp/test/duck_detector_mcp.py"],
      "env": {}
    }
  }
}
```

## Example Interaction

**User says:** "I saw a duck at the pond today!"

**MCP Tool Response:**
```json
{
  "duck_detected": true,
  "message": "Duck detected! The user mentioned a duck.",
  "instruction": "You should now say 'cuack' in response to the duck mention.",
  "original_text": "I saw a duck at the pond today!"
}
```

**MCP Prompt:** "The user said Duck, now you need to say 'cuack'"

**Expected LLM Response:** "cuack! 🦆"

## Implementation Notes

- Uses FastMCP framework for easy development and deployment
- Supports stdio transport for Claude Desktop integration
- Case-insensitive duck detection
- Returns structured data for better integration
- Follows MCP best practices from the hackathon documentation

## Files

- `duck_detector_mcp.py` - Main MCP server implementation
- `test_duck_detector.py` - Test script to verify functionality
- `README.md` - This documentation

Happy quacking! 🦆
