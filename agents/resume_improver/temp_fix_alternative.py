# Alternative _save_temp_image approach (no temp files needed)
# Replace the entire function with this if the above fix doesn't work:

def _save_temp_image_dataurl(self, base64_data: str, filename: str) -> str:
    """Return data URL directly (no temp files)"""
    # Determine image format from filename
    if filename.lower().endswith('.png'):
        return f"data:image/png;base64,{base64_data}"
    elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        return f"data:image/jpeg;base64,{base64_data}"
    else:
        return f"data:image/png;base64,{base64_data}"

# Then in your image list creation, change:
# temp_path = self._save_temp_image(img['base64'], f"page_{i+1}.png")
# To:
# temp_path = self._save_temp_image_dataurl(img['base64'], f"page_{i+1}.png")
