# BULLETPROOF FIX: Replace the image list creation sections with this
# This approach uses data URLs directly, avoiding temp files entirely

# Replace lines 110-115 (page images) with:
page_image_list = []
for i, img in enumerate(page_images):
    data_url = f"data:image/png;base64,{img['base64']}"
    page_image_list.append((data_url, f"Page {img['page_number']} ({img['width']}x{img['height']})"))

# Replace lines 117-125 (embedded images) with:
embedded_image_list = []
for i, img in enumerate(embedded_images):
    headshot_flag = "🖼️ Potential Headshot" if img['is_potential_headshot'] else "🖼️ Image"
    caption = f"{headshot_flag} - Page {img['page_number']} ({img['width']}x{img['height']}, {img['format']})"
    data_url = f"data:image/{img['format'].lower()};base64,{img['base64']}"
    embedded_image_list.append((data_url, caption))

# And remove the _save_temp_image function entirely since it's not needed
