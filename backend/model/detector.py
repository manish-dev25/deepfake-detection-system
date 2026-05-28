# detector.py — Hugging Face Inference API (no torch/transformers needed)
import requests
from PIL import Image
import io

API_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
API_TOKEN = "" 

def predict(image: Image.Image) -> dict:
    image = image.convert("RGB")
    
    # Image ko bytes mein convert karo
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, data=img_bytes)
    results = response.json()

    ai_score = 0.0
    real_score = 0.0

    for item in results:
        label = item["label"].lower()
        score = round(item["score"] * 100, 2)
        if label in ["artificial", "fake", "ai"]:
            ai_score = score
        elif label in ["human", "real"]:
            real_score = score

    is_ai = ai_score > real_score
    verdict = "AI Generated" if is_ai else "Real Image"
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