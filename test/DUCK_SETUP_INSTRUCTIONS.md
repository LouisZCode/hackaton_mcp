# Duck Detector MCP - Claude Desktop Setup

## Files Created

1. **`claude_desktop_duck_config.json`** - Just the duck detector config
2. **`claude_desktop_config_example.json`** - Complete example with filesystem + duck detector

## Setup Instructions

### Step 1: Locate Your Claude Desktop Config
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

### Step 2: Add Duck Detector to Your Config

**Option A - New Config:**
Copy the contents of `claude_desktop_config_example.json` and replace `[YOUR_USERNAME]` with your actual username.

**Option B - Add to Existing Config:**
Add this entry to your existing `mcpServers` section:
```json
"duck-detector": {
  "command": "C:/projects/GitHub/hackaton_mcp/.venv/Scripts/python.exe",
  "args": ["C:/projects/GitHub/hackaton_mcp/test/duck_detector_mcp.py"],
  "env": {}
}
```

### Step 3: Restart Claude Desktop
Close and reopen Claude Desktop for changes to take effect.

### Step 4: Verify Installation
1. Look for the slider icon (🔧) in the bottom left of the input box
2. Click it to see available tools
3. You should see "detect_duck_in_text" tool

## Testing the Duck Detector

Try these phrases in Claude Desktop:
- "I saw a duck today"
- "Look at that rubber duck"
- "DUCK hunting season"

Claude should:
1. Automatically detect the word "duck" 
2. Use the MCP tool to analyze the text
3. Respond with "cuack" as instructed by the MCP prompts

## Troubleshooting

**If the server doesn't appear:**
1. Check that Python path exists: `C:/projects/GitHub/hackaton_mcp/.venv/Scripts/python.exe`
2. Verify the MCP file exists: `C:/projects/GitHub/hackaton_mcp/test/duck_detector_mcp.py`
3. Test the server manually: Run the python file directly (should be silent)
4. Check JSON syntax is valid
5. Restart Claude Desktop

**Based on the key learnings from the Notion documentation:**
- ✅ Using full Windows paths (C:/ format)
- ✅ Pointing to virtual environment python.exe
- ✅ Using empty env object
- ✅ Following working config pattern
