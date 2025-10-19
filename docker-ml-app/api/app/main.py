# docker-ml-app/api/app/main.py
import io
from typing import List

import torch
import torchvision as tv
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from PIL import Image

app = FastAPI(title="ResNet18 Inference API", version="1.0")

# ---- Load model & transforms once (on startup) ----
weights = tv.models.ResNet18_Weights.DEFAULT
model = tv.models.resnet18(weights=weights).eval()
preprocess = weights.transforms()
classes: List[str] = list(weights.meta["categories"])

# If GPU is available inside the container (optional), use it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/infer")
async def infer(
    file: UploadFile = File(..., description="RGB image file (jpg/png)"),
    topk: int = Query(5, ge=1, le=100, description="Top-K predictions"),
):
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    batch = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[0]
        top = torch.topk(probs, k=min(topk, probs.shape[0]))

    results = [
        {"label": classes[idx], "prob": float(prob)}
        for prob, idx in zip(top.values.cpu().tolist(), top.indices.cpu().tolist())
    ]
    return {"topk": results}
