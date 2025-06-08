
import os

# we used   uv add mcp[cli] httpx   to get these:
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("emotion-face-detector")

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


#let's add our helper functions:

async def detect_emotion_from_text_internal(text: str) -> dict:
    """Internal helper to detect emotion and return structured data"""
    # Enhanced emotion detection with more emotions
    if any(word in text.lower() for word in ['happy', 'joy', 'good', 'great', 'wonderful', 'amazing', 'excited', 'love']):
        emotion = "happy"
        response_text = f"Detected emotion: Happy - This text expresses positive feelings!"
    elif any(word in text.lower() for word in ['angry', 'mad', 'furious', 'hate', 'rage', 'annoyed']):
        emotion = "angry"
        response_text = f"Detected emotion: Angry - This text expresses anger or frustration."
    elif any(word in text.lower() for word in ['surprised', 'wow', 'shocked', 'amazing', 'incredible', 'unbelievable']):
        emotion = "surprised"
        response_text = f"Detected emotion: Surprised - This text expresses surprise or amazement."
    elif any(word in text.lower() for word in ['sad', 'bad', 'terrible', 'awful', 'depressed', 'crying', 'upset']):
        emotion = "sad"
        response_text = f"Detected emotion: Sad - This text expresses negative feelings."
    else:
        emotion = "neutral"
        response_text = f"Detected emotion: Neutral - This text has neutral emotional tone."
    
    return {
        "emotion": emotion,
        "text": response_text,
        "image_file": f"{emotion}.png"
    }


@mcp.tool()
async def detect_emotion_from_text(text: str) -> str:
    """Detect emotion from input text and return analysis

    Args:
        text: Input text to analyze for emotional content
    """
    result = await detect_emotion_from_text_internal(text)
    return result["text"]


@mcp.prompt()
async def emotion_prompt(emotion: str) -> str:
    """Get a simple explanation prompt for each detected emotion
    
    Args:
        emotion: The emotion type (happy, sad, angry, surprised, neutral)
    """
    prompts = {
        "happy": "This person is feeling joy and positivity. They're in a good mood and expressing contentment.",
        "sad": "This person is experiencing sadness or disappointment. They may need comfort or understanding.",
        "angry": "This person is feeling frustrated or upset. They're expressing anger or irritation about something.",
        "surprised": "This person is amazed or shocked by something unexpected. They're experiencing wonder or disbelief.",
        "neutral": "This person has a calm, balanced emotional state. No strong emotions are being expressed."
    }
    
    return prompts.get(emotion.lower(), "Unknown emotion detected.")


if __name__ == "__main__":
    # Run as pure MCP server for Claude Desktop
    mcp.run(transport='stdio')