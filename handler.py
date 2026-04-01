import runpod
import base64
import os
import damage_service

# Initialize models (if needed) at startup
# YOLO and Gemma are handled by damage_service.py

def handler(job):
    """
    RunPod Serverless Handler for Vehicle Damage Vision API.
    Input Format:
    {
        "image_base64": "...",
        "label": "Vehicle Side"
    }
    """
    job_input = job["input"]
    image_b64 = job_input.get("image_base64")
    label     = job_input.get("label", "Vehicle")

    if not image_b64:
        return {"error": "Missing image_base64 in input"}

    try:
        # Decode the base64 string to bytes
        # Remove header if present (e.g. data:image/jpeg;base64,...)
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        return {"error": f"Failed to decode base64 image: {str(e)}"}

    # Process the image using the existing service
    try:
        result = damage_service.assess_damage(image_bytes, image_label=label)
        
        # result contains "report" (dict) and "annotated_image_bytes" (bytes)
        # We need to convert annotated_image_bytes to base64 for JSON output
        report = result.get("report", {})
        ann_bytes = result.get("annotated_image_bytes")
        
        ann_b64 = ""
        if ann_bytes:
            ann_b64 = base64.b64encode(ann_bytes).decode('ascii')
            
        return {
            "report": report,
            "annotated_image_base64": ann_b64
        }
        
    except Exception as e:
        return {"error": f"Internal assessment error: {str(e)}"}

# Start the RunPod Serverless worker
runpod.serverless.start({"handler": handler})
