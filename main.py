from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import uvicorn

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)

def generate_text(image: Image.Image) -> str:
    prompt = "Question: Which object is being shown? Answer:"
    inputs = processor(images = image, text=prompt, return_tensors="pt").to(device, torch.float16)
    #inputs = processor(images = image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

@app.post("/generate_text/")
async def upload_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    generated_text = generate_text(image)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, workers=1)
