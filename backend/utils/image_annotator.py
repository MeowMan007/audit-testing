import base64
import io
import logging
from typing import List, Dict, Any
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

def annotate_screenshot(screenshot_b64: str, issues: List[Any], element_rects: Dict[str, Dict[str, float]]) -> str:
    """
    Draws red bounding boxes on the base64 screenshot for each issue's element.
    Returns the annotated screenshot as a base64 string.
    """
    if not screenshot_b64 or not element_rects:
        return screenshot_b64
        
    try:
        # Decode base64 to image
        image_data = base64.b64decode(screenshot_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        
        # Create a transparent overlay for drawing
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        drawn_rects = 0
        
        # Draw bounding boxes for each issue
        for issue in issues:
            data_al_id = getattr(issue, "data_al_id", None)
            if data_al_id and data_al_id in element_rects:
                rect = element_rects[data_al_id]
                x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
                
                # Draw a semi-transparent red filled rectangle
                fill_color = (255, 0, 0, 40)  # Red with alpha
                outline_color = (255, 0, 0, 255) # Solid red
                
                # Bounding box coordinates: [x0, y0, x1, y1]
                bbox = [x, y, x + w, y + h]
                
                # Draw filled rectangle and outline
                draw.rectangle(bbox, fill=fill_color, outline=outline_color, width=3)
                drawn_rects += 1
                
        if drawn_rects > 0:
            # Composite the overlay onto the original image
            final_image = Image.alpha_composite(image, overlay)
            final_image = final_image.convert("RGB") # Remove alpha for JPEG/PNG
            
            # Encode back to base64
            buffer = io.BytesIO()
            final_image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        return screenshot_b64
        
    except Exception as e:
        logger.error(f"Failed to annotate screenshot: {e}")
        return screenshot_b64
