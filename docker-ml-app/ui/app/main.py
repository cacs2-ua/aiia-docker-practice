# docker-ml-app/ui/app/main.py
import os
import io
import json
import requests

# Hard-disable extra Gradio behaviors that can break in containers
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["GRADIO_SHOW_API"] = "false"

import gradio as gr
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
INFER_URL = f"{API_URL}/infer"
HEALTH_URL = f"{API_URL}/health"

def check_api():
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        r.raise_for_status()
        return f"OK → {r.json()}"
    except Exception as e:
        return f"ERROR → {e}"

def call_infer(pil_img, topk=5):
    if pil_img is None:
        return [[1, "no image", 0.0]]
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)
    files = {"file": ("image.jpg", buf, "image/jpeg")}
    params = {"topk": int(topk)}
    resp = requests.post(INFER_URL, files=files, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("topk", [])
    table = []
    for i, r in enumerate(rows, start=1):
        table.append([i, r.get("label", ""), round(float(r.get("prob", 0.0)), 6)])
    return table

with gr.Blocks(title="ResNet18 GUI", analytics_enabled=False) as demo:
    gr.Markdown(
        "# 🖼️ ResNet18 Image Classification (Gradio)\n"
        f"- Backend API: **{API_URL}**\n"
        "- Upload an image and get Top-K predictions."
    )

    with gr.Row():
        api_status = gr.Textbox(label="API status", interactive=False)
        gr.Button("Check API health").click(fn=check_api, outputs=[api_status])

    with gr.Row():
        img = gr.Image(type="pil", label="Input image")
        topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")

    results = gr.Dataframe(
        headers=["rank", "label", "prob"],
        datatype=["number", "str", "number"],
        row_count=5,
        label="Predictions",
    )

    # Use a lambda (no type annotations) to avoid schema introspection edge-cases
    gr.Button("Run inference").click(
        fn=lambda im, k: call_infer(im, k),
        inputs=[img, topk],
        outputs=[results],
    )

if __name__ == "__main__":
    # Bind to all interfaces for Docker; don't auto-open a browser
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False, inbrowser=False)
