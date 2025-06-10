"""
Flux Kontext API integration for professional LinkedIn image enhancement
Based on the flux_kontext_example.py implementation
"""

import os
import io
import base64
import time
import logging
from typing import Optional, Dict, Any
from PIL import Image
import requests

# Setup logging
logger = logging.getLogger(__name__)

class FluxImageEnhancer:
    """Professional image enhancement using Flux Kontext API"""
    
    def __init__(self):
        self.api_key = os.environ.get("BFL_API_KEY")
        self.base_url = "https://api.bfl.ai/v1"
        self.professional_prompt = (
            "Make the person wear a light blue blazer, make the background white and clean any noise in the foreground. "
            "make the hair more orderly. Keep the face of the person intact. keep the gender of the person intact. "
            "the image should always be a bust"
        )
        
        if not self.api_key:
            logger.warning("BFL_API_KEY not found in environment variables")
    
    async def enhance_image_async(self, image: Image.Image, custom_prompt: str = None) -> Optional[Image.Image]:
        """
        Asynchronously enhance an image using Flux Kontext API
        
        Args:
            image: PIL Image to enhance
            custom_prompt: Optional custom prompt (uses professional LinkedIn prompt by default)
            
        Returns:
            Enhanced PIL Image or None if failed
        """
        if not self.api_key:
            logger.error("BFL_API_KEY not configured")
            return None
        
        try:
            # Use professional prompt by default
            prompt = custom_prompt if custom_prompt else self.professional_prompt
            
            # Encode image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info(f"Submitting image enhancement request with prompt: {prompt[:50]}...")
            
            # Submit enhancement request
            response = requests.post(
                f'{self.base_url}/flux-kontext-pro',
                headers={
                    'accept': 'application/json',
                    'x-key': self.api_key,
                    'Content-Type': 'application/json',
                },
                json={
                    'prompt': prompt,
                    'input_image': img_str,
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Flux API request failed: {response.status_code} - {response.text}")
                return None
                
            request_data = response.json()
            request_id = request_data.get("id")
            
            if not request_id:
                logger.error("No request ID received from Flux API")
                return None
                
            logger.info(f"Enhancement request submitted successfully. Request ID: {request_id}")
            
            # Poll for result
            enhanced_image = await self._poll_for_result(request_id)
            
            if enhanced_image:
                logger.info("Image enhancement completed successfully")
                return enhanced_image
            else:
                logger.error("Image enhancement failed or timed out")
                return None
                
        except Exception as e:
            logger.error(f"Error during image enhancement: {e}")
            return None
    
    async def _poll_for_result(self, request_id: str, max_attempts: int = 30, poll_interval: int = 2) -> Optional[Image.Image]:
        """
        Poll Flux API for enhancement result
        
        Args:
            request_id: Request ID from initial submission
            max_attempts: Maximum polling attempts
            poll_interval: Seconds between polling attempts
            
        Returns:
            Enhanced PIL Image or None if failed
        """
        logger.info(f"Starting polling for request {request_id} (max {max_attempts} attempts)")
        
        for attempt in range(max_attempts):
            try:
                # Add delay before polling
                if attempt > 0:
                    time.sleep(poll_interval)
                
                # Check result
                result_response = requests.get(
                    f'{self.base_url}/get_result?id={request_id}',
                    headers={
                        'accept': 'application/json',
                        'x-key': self.api_key,
                    },
                    timeout=15
                )
                
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    status = result_data.get("status")
                    
                    logger.debug(f"Polling attempt {attempt + 1}: Status = {status}")
                    
                    if status == "Ready":
                        # Enhancement completed successfully
                        image_url = result_data.get("result", {}).get("sample")
                        if image_url:
                            # Download the enhanced image
                            img_response = requests.get(image_url, timeout=30)
                            if img_response.status_code == 200:
                                enhanced_image = Image.open(io.BytesIO(img_response.content))
                                logger.info(f"Successfully downloaded enhanced image after {attempt + 1} attempts")
                                return enhanced_image
                            else:
                                logger.error(f"Failed to download enhanced image: {img_response.status_code}")
                                return None
                        else:
                            logger.error("No image URL in ready response")
                            return None
                            
                    elif status == "Error":
                        error_msg = result_data.get('result', 'Unknown error')
                        logger.error(f"Flux enhancement failed with error: {error_msg}")
                        return None
                        
                    elif status in ["Pending", "Processing"]:
                        # Continue polling
                        continue
                        
                    else:
                        logger.warning(f"Unknown status received: {status}")
                        continue
                        
                else:
                    logger.warning(f"Polling attempt {attempt + 1} failed: {result_response.status_code}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Polling attempt {attempt + 1} error: {e}")
                continue
        
        logger.error(f"Image enhancement timed out after {max_attempts} attempts")
        return None
    
    def enhance_image_sync(self, image: Image.Image, custom_prompt: str = None) -> Optional[Image.Image]:
        """
        Synchronously enhance an image (wrapper for async method)
        
        Args:
            image: PIL Image to enhance
            custom_prompt: Optional custom prompt
            
        Returns:
            Enhanced PIL Image or None if failed
        """
        import asyncio
        
        try:
            # Run the async enhancement
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.enhance_image_async(image, custom_prompt))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Sync enhancement wrapper failed: {e}")
            return None
    
    def optimize_for_linkedin(self, enhanced_image: Image.Image, target_size: tuple = (800, 800)) -> Image.Image:
        """
        Optimize enhanced image for LinkedIn profile picture requirements
        
        Args:
            enhanced_image: Enhanced image from Flux
            target_size: Target dimensions (width, height)
            
        Returns:
            LinkedIn-optimized PIL Image
        """
        try:
            target_width, target_height = target_size
            
            # Calculate aspect ratio
            original_width, original_height = enhanced_image.size
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
            img_resized = enhanced_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a new image with LinkedIn dimensions and white background
            linkedin_img = Image.new('RGB', (target_width, target_height), 'white')
            
            # Calculate position to center the resized image
            x = (target_width - new_width) // 2
            y = (target_height - new_height) // 2
            
            # Paste the resized image onto the LinkedIn-sized canvas
            linkedin_img.paste(img_resized, (x, y))
            
            logger.info(f"Image optimized for LinkedIn: {target_width}x{target_height}")
            return linkedin_img
            
        except Exception as e:
            logger.error(f"LinkedIn optimization failed: {e}")
            return enhanced_image  # Return original if optimization fails
    
    def enhance_for_linkedin_complete(self, image: Image.Image, custom_prompt: str = None) -> Optional[Image.Image]:
        """
        Complete LinkedIn enhancement workflow: Flux enhancement + LinkedIn optimization
        
        Args:
            image: Original PIL Image
            custom_prompt: Optional custom prompt
            
        Returns:
            Final LinkedIn-optimized image or None if failed
        """
        try:
            logger.info("Starting complete LinkedIn enhancement workflow")
            
            # Step 1: Enhance with Flux Kontext
            enhanced_image = self.enhance_image_sync(image, custom_prompt)
            
            if not enhanced_image:
                logger.error("Flux enhancement failed, cannot proceed with LinkedIn optimization")
                return None
            
            # Step 2: Optimize for LinkedIn
            linkedin_optimized = self.optimize_for_linkedin(enhanced_image)
            
            logger.info("Complete LinkedIn enhancement workflow finished successfully")
            return linkedin_optimized
            
        except Exception as e:
            logger.error(f"Complete LinkedIn enhancement failed: {e}")
            return None

# Convenience functions
def enhance_image_for_linkedin(image: Image.Image, custom_prompt: str = None) -> Optional[Image.Image]:
    """
    Convenience function to enhance an image for LinkedIn
    
    Args:
        image: PIL Image to enhance
        custom_prompt: Optional custom prompt
        
    Returns:
        LinkedIn-optimized image or None if failed
    """
    enhancer = FluxImageEnhancer()
    return enhancer.enhance_for_linkedin_complete(image, custom_prompt)

def is_flux_configured() -> bool:
    """Check if Flux Kontext API is properly configured"""
    return bool(os.environ.get("BFL_API_KEY"))