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


def annotate_reading_order(screenshot_b64: str, visual_order_map: List[Dict[str, Any]]) -> str:
    """
    Overlay numbered circles on the screenshot showing reading order.
    Draws DOM order numbers, and if there's a drift, draws the visual order too.
    """
    if not screenshot_b64 or not visual_order_map:
        return screenshot_b64
        
    try:
        image_data = base64.b64decode(screenshot_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        drawn = 0
        for elem in visual_order_map:
            bbox = elem.get('bbox', {})
            if not bbox: continue
            
            x, y = bbox.get('x', 0), bbox.get('y', 0)
            dom_rank = elem.get('dom_rank', 0)
            visual_rank = elem.get('visual_rank', 0)
            drift = elem.get('drift', 0)
            
            # Draw DOM order circle (Green if match, Orange/Red if mismatch)
            r = 14
            cx, cy = x + r, y + r
            
            if drift >= 10:
                fill_color = (244, 67, 54, 200) # Red
            elif drift >= 5:
                fill_color = (255, 152, 0, 200) # Orange
            else:
                fill_color = (76, 175, 80, 180) # Green
                
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill_color, outline=(255,255,255,255), width=2)
            
            # Try to load a font, otherwise use default
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("arial.ttf", 14)
            except Exception:
                font = None
                
            # Draw DOM text
            text = str(dom_rank)
            # Center text roughly
            draw.text((cx-5 if len(text) == 1 else cx-9, cy-7), text, fill=(255,255,255,255), font=font)
            
            # If significant drift, draw visual rank below it
            if drift >= 5:
                v_text = f"v{visual_rank}"
                draw.text((cx-r, cy+r+2), v_text, fill=(255, 152, 0, 255), font=font)
                
            drawn += 1
            
        if drawn > 0:
            final_image = Image.alpha_composite(image, overlay)
            final_image = final_image.convert("RGB")
            buffer = io.BytesIO()
            final_image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        return screenshot_b64
        
    except Exception as e:
        logger.error(f"Failed to annotate reading order: {e}")
        return screenshot_b64
