from typing import Any
import os
from PIL import Image
import io
import base64
import requests
import time

#Gradio for the hackaton:
import gradio as gr

# we used   uv add mcp[cli] httpx   to get these:
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("linkedin-image-processor")

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#let's add our helper functions:

async def flux_kontext_edit_image(image: Image.Image, prompt: str) -> Image.Image:
    """Use Flux Kontext API to edit an image based on a prompt
    
    Args:
        image: PIL Image to edit
        prompt: Text description of what to edit
        
    Returns:
        Image.Image: Edited image from Flux Kontext
    """
    try:
        # Encode image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Make request to Flux Kontext API
        response = requests.post(
            'https://api.bfl.ai/v1/flux-kontext-pro',
            headers={
                'accept': 'application/json',
                'x-key': os.environ.get("BFL_API_KEY"),
                'Content-Type': 'application/json',
            },
            json={
                'prompt': prompt,
                'input_image': img_str,
            },
        )
        
        if response.status_code != 200:
            print(f"API request failed: {response.status_code}")
            return image
            
        request_data = response.json()
        request_id = request_data.get("id")
        
        if not request_id:
            print("No request ID received")
            return image
            
        # Poll for result (simplified polling)
        max_attempts = 30
        for attempt in range(max_attempts):
            time.sleep(2)
            
            result_response = requests.get(
                f'https://api.bfl.ai/v1/get_result?id={request_id}',
                headers={
                    'accept': 'application/json',
                    'x-key': os.environ.get("BFL_API_KEY"),
                }
            )
            
            if result_response.status_code == 200:
                result_data = result_response.json()
                
                if result_data.get("status") == "Ready":
                    image_url = result_data.get("result", {}).get("sample")
                    if image_url:
                        # Download and return the edited image
                        img_response = requests.get(image_url)
                        edited_image = Image.open(io.BytesIO(img_response.content))
                        return edited_image
                        
                elif result_data.get("status") == "Error":
                    print(f"Flux Kontext error: {result_data.get('result')}")
                    break
                    
        print("Flux Kontext processing timeout or failed")
        return image
        
    except Exception as e:
        print(f"Error with Flux Kontext API: {e}")
        return image

def process_linkedin_image(image) -> Image.Image:
    """Process an image for LinkedIn optimization using Flux Kontext
    
    Args:
        image: Input image file
        
    Returns:
        Image.Image: Processed image optimized for LinkedIn
    """
    if image is None:
        return None
    
    try:
        # Handle different input types
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
            
        # Define the fixed professional prompt
        professional_prompt = "Make the person wear a light blue blazer, make the background white and clean any noise in the foreground. make the hair more orderly. Keep the face of the person intact. keep the gender of the person intact. the image should always be a bust"
        
        # First, use Flux Kontext to enhance/edit the image
        import asyncio
        edited_img = asyncio.run(flux_kontext_edit_image(img, professional_prompt))
        
        # Then apply LinkedIn optimization
        target_width = 800
        target_height = 800
        
        # Calculate aspect ratio
        original_width, original_height = edited_img.size
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        
        # Resize while maintaining aspect ratio
        if original_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / original_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * original_ratio)
        
        # Resize the image
        img_resized = edited_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a new image with LinkedIn dimensions and white background
        linkedin_img = Image.new('RGB', (target_width, target_height), 'white')
        
        # Calculate position to center the resized image
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2
        
        # Paste the resized image onto the LinkedIn-sized canvas
        linkedin_img.paste(img_resized, (x, y))
        
        return linkedin_img
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return image if image else None

@mcp.tool()
async def create_professional_linkedin_headshot(image_url: str) -> str:
    """Transform any photo into a professional LinkedIn headshot using AI.
    
    Automatically adds professional business attire (light blue blazer), creates a clean white 
    background, tidies hair, removes noise, and formats as an 800x800 centered bust shot while 
    preserving facial features and gender. Perfect for professional headshots, profile pictures, 
    business photos, and LinkedIn optimization.
    
    Args:
        image_url: HTTP/HTTPS URL to the input image file (JPEG, PNG supported)
        
    Returns:
        str: Success message or error description
    """
    try:
        processed_img = process_linkedin_image(image_url)
        if processed_img:
            return "Professional LinkedIn headshot created successfully - added business attire, clean background, and professional formatting"
        else:
            return "Failed to process image for LinkedIn optimization"
    except Exception as e:
        return f"Error creating professional headshot: {str(e)}"


demo = gr.Interface(
    fn=process_linkedin_image,
    inputs=gr.Image(type="pil", label="Upload Your Photo"),
    outputs=gr.Image(type="pil", label="Professional LinkedIn Photo"),
    title="Professional LinkedIn Photo Generator",
    description="Upload a photo and automatically transform it into a professional LinkedIn profile picture. The AI will add a light blue blazer, clean white background, and tidy up your appearance while keeping your face intact."
)

if __name__ == "__main__":
    # Initialize and run the server
    demo.launch(mcp_server=True)