from typing import Any
import os
import base64
import requests
import time

# we used   uv add mcp[cli] httpx   to get these:
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("linkedin-image-processor")

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


#let's add our helper functions:

async def flux_kontext_edit_image_internal(image_url: str, prompt: str) -> dict:
    """Internal helper to process image with Flux Kontext API"""
    # Check for API key
    api_key = os.environ.get("BFL_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "BFL_API_KEY environment variable not set",
            "result_url": None
        }
    
    try:
        # Download the input image
        img_response = requests.get(image_url, timeout=30)
        if img_response.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to download image: {img_response.status_code}",
                "result_url": None
            }
        
        # Encode image to base64
        img_str = base64.b64encode(img_response.content).decode()
        
        # Make request to Flux Kontext API
        response = requests.post(
            'https://api.bfl.ai/v1/flux-kontext-pro',
            headers={
                'accept': 'application/json',
                'x-key': api_key,
                'Content-Type': 'application/json',
            },
            json={
                'prompt': prompt,
                'input_image': img_str,
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API request failed: {response.status_code} - {response.text}",
                "result_url": None
            }
            
        request_data = response.json()
        request_id = request_data.get("id")
        
        if not request_id:
            return {
                "success": False,
                "error": "No request ID received from API",
                "result_url": None
            }
            
        # Poll for result
        max_attempts = 30
        for attempt in range(max_attempts):
            time.sleep(2)
            
            result_response = requests.get(
                f'https://api.bfl.ai/v1/get_result?id={request_id}',
                headers={
                    'accept': 'application/json',
                    'x-key': api_key,
                },
                timeout=10
            )
            
            if result_response.status_code == 200:
                result_data = result_response.json()
                
                if result_data.get("status") == "Ready":
                    image_url = result_data.get("result", {}).get("sample")
                    if image_url:
                        return {
                            "success": True,
                            "error": None,
                            "result_url": image_url
                        }
                        
                elif result_data.get("status") == "Error":
                    return {
                        "success": False,
                        "error": f"Flux Kontext error: {result_data.get('result')}",
                        "result_url": None
                    }
                    
        return {
            "success": False,
            "error": "Processing timeout - try again later",
            "result_url": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error with Flux Kontext API: {str(e)}",
            "result_url": None
        }


@mcp.tool()
async def create_professional_linkedin_headshot(image_url: str) -> str:
    """Transform any photo into a professional LinkedIn headshot using AI.
    
    Automatically adds professional business attire (light blue blazer), creates a clean white 
    background, tidies hair, removes noise, and formats as a centered bust shot while 
    preserving facial features and gender. Perfect for professional headshots, profile pictures, 
    business photos, and LinkedIn optimization.
    
    Args:
        image_url: HTTP/HTTPS URL to the input image file (JPEG, PNG supported)
        
    Returns:
        str: Success message with result URL or error description
    """
    # Define the professional prompt
    professional_prompt = (
        "Make the person wear a light blue blazer, make the background white and clean "
        "any noise in the foreground. Make the hair more orderly. Keep the face of the "
        "person intact. Keep the gender of the person intact. The image should always be a bust."
    )
    
    # Process the image
    result = await flux_kontext_edit_image_internal(image_url, professional_prompt)
    
    if result["success"]:
        return f"Professional LinkedIn headshot created successfully! Result URL: {result['result_url']}"
    else:
        return f"Failed to create professional headshot: {result['error']}"


@mcp.prompt()
async def linkedin_photo_tips(style: str) -> str:
    """Get professional photo tips for LinkedIn optimization
    
    Args:
        style: The style type (professional, creative, executive, casual, technical)
    """
    tips = {
        "professional": "For professional LinkedIn photos: Use good lighting, maintain eye contact with camera, wear business attire, keep background clean and simple, smile naturally, and ensure high image quality.",
        "creative": "For creative LinkedIn photos: Show personality while staying professional, use interesting but not distracting backgrounds, incorporate relevant props subtly, maintain approachable expression.",
        "executive": "For executive LinkedIn photos: Project authority and confidence, use formal business attire, maintain serious but approachable expression, ensure impeccable grooming, use neutral backgrounds.",
        "casual": "For casual professional photos: Smart casual attire, relaxed but professional posture, natural smile, clean background, show approachability while maintaining professionalism.",
        "technical": "For technical professional photos: Clean, modern appearance, possibly include subtle tech elements, maintain professional but innovative look, clear and crisp image quality."
    }
    
    return tips.get(style.lower(), "For any LinkedIn photo: Focus on professionalism, good lighting, clean background, appropriate attire, and natural expression.")


if __name__ == "__main__":
    # Run as pure MCP server for Claude Desktop
    mcp.run(transport='stdio')