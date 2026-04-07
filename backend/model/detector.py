# detector.py
# CNN Model loader and predictor

from transformers import pipeline
from PIL import Image
import torch

# ─── Load model once when server starts ───
print("Loading CNN model from Hugging Face...")

detector_pipeline = pipeline(
    "image-classification",
    model="umm-maybe/AI-image-detector",
    device=0 if torch.cuda.is_available() else -1  
    # 0 = GPU, -1 = CPU
)

print("Model loaded successfully!")


def predict(image: Image.Image) -> dict:
    """
    Takes a PIL Image
    Returns prediction dict with scores
    """
    # Model expects RGB
    image = image.convert("RGB")
    
    # Run inference
    results = detector_pipeline(image)
    
    # results looks like:
    # [{'label': 'artificial', 'score': 0.874},
    #  {'label': 'human', 'score': 0.126}]
    
    ai_score = 0.0
    real_score = 0.0
    
    for item in results:
        label = item["label"].lower()
        score = round(item["score"] * 100, 2)
        
        if label in ["artificial", "fake", "ai"]:
            ai_score = score
        elif label in ["human", "real"]:
            real_score = score
    
    # Determine verdict
    is_ai = ai_score > real_score
    verdict = "AI Generated" if is_ai else "Real Image"
    
    # Confidence level
    top_score = max(ai_score, real_score)
    if top_score >= 80:
        confidence = "HIGH"
    elif top_score >= 55:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return {
        "verdict": verdict,
        "ai_score": ai_score,
        "real_score": real_score,
        "confidence": confidence,
        "is_ai": is_ai
    }