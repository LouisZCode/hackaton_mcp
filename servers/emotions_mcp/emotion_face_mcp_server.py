from typing import Any, Tuple
from PIL import Image
import os

#Gradio for the hackaton:
import gradio as gr

# we used   uv add mcp[cli] httpx   to get these:
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("emotion-face-detector")

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


#let's add our helper functions:

async def generate_emotion_response(text: str) -> Tuple[Image.Image, str]:
    """Generate an emotion image and response text based on input text"""
    # Enhanced emotion detection with more emotions
    if any(word in text.lower() for word in ['happy', 'joy', 'good', 'great', 'wonderful', 'amazing', 'excited', 'love']):
        emotion = "Happy"
        image_file = "happy.png"
        response_text = f"Detected emotion: {emotion} - This text expresses positive feelings!"
    elif any(word in text.lower() for word in ['angry', 'mad', 'furious', 'hate', 'rage', 'annoyed']):
        emotion = "Angry"
        image_file = "angry.png"
        response_text = f"Detected emotion: {emotion} - This text expresses anger or frustration."
    elif any(word in text.lower() for word in ['surprised', 'wow', 'shocked', 'amazing', 'incredible', 'unbelievable']):
        emotion = "Surprised"
        image_file = "surprise.png"
        response_text = f"Detected emotion: {emotion} - This text expresses surprise or amazement."
    elif any(word in text.lower() for word in ['sad', 'bad', 'terrible', 'awful', 'depressed', 'crying', 'upset']):
        emotion = "Sad"
        image_file = "sad.png"
        response_text = f"Detected emotion: {emotion} - This text expresses negative feelings."
    else:
        emotion = "Neutral"
        image_file = "neutral.png"
        response_text = f"Detected emotion: {emotion} - This text has neutral emotional tone."
    
    # Load image from local folder
    image_path = os.path.join(SCRIPT_DIR, image_file)
    try:
        img = Image.open(image_path)
        # Resize if too large (over 300KB files might be too big for web interface)
        if img.size[0] > 512 or img.size[1] > 512:
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
    except FileNotFoundError:
        # Fallback: create a simple placeholder if image not found
        img = Image.new('RGB', (400, 400), color='lightgray')
        response_text += f" (Image {image_file} not found - using placeholder)"
    
    return img, response_text



@mcp.tool()
async def detect_emotion_from_text(text: str) -> str:
    """Detect emotion from input text and generate visual representation

    Args:
        text: Input text to analyze for emotional content
    """
    _, response_text = await generate_emotion_response(text)
    return response_text


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


demo = gr.Interface(
    fn=generate_emotion_response,
    inputs=gr.Textbox(placeholder="Enter text to analyze emotions", label="Input Text"),
    outputs=[
        gr.Image(type="pil", label="Emotion Visualization"),
        gr.Textbox(label="Emotion Analysis")
    ],
    title="Emotion Prompt Creation MCP Server",
    description="Analyze text for emotional content and generate Prompt + visual representation"
)

if __name__ == "__main__":
    # Initialize and run the server
    demo.launch(mcp_server=True)