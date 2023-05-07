from fastapi import FastAPI, File, UploadFile
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from GroundingDINO.groundingdino.util import box_ops
from PIL import Image
import io
import cv2
import numpy as np
import uvicorn
from fastapi import Form
import tempfile
import os
from starlette.responses import FileResponse
import torch
from segment_anything import SamPredictor, sam_model_registry
import supervision as sv
import imageio
from transformers import CLIPProcessor, CLIPModel
import pinecone
import random


app = FastAPI()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load GroundingDino model
model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth").to(device=DEVICE)
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(DEVICE)
predictor = SamPredictor(sam)

# Load CLIP model
model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
@app.post("/annotate")
async def annotate_image(text_prompt: str, photo: UploadFile = File(...)):
    
    # Zero-Shot object detection (GroundingDINO)
    input_data = text_prompt
    image_bytes = io.BytesIO(await photo.read())
    image_np = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    
    output_filename = "original_image1.png"
    cv2.imwrite(output_filename, image)

    image_source, image_dino = load_image("original_image1.png")

    boxes, logits, phrases = predict(
        model=model,
        image=image_dino,
        caption=input_data,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    output_filename = "annotated_image.jpg"
    cv2.imwrite(output_filename, annotated_frame)
    
    
    # Zero-Shot object segmentation (SAM)
    predictor.set_image(image)
    H, W ,_ = image.shape

    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy.to(DEVICE), image.shape[:2])
    
    boxes_xyxy = boxes_xyxy.numpy()[0].astype(int)
    transformed_boxes = transformed_boxes.cpu().numpy()[0].astype(int)
    
    masks, scores, logits = predictor.predict(box=boxes_xyxy, multimask_output=True)
    
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks),mask=masks)
    detections = detections[detections.area == np.max(detections.area)]

    segmented_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    output_filename = "segmented_image.jpg"
    cv2.imwrite(output_filename, segmented_image)
    
    # Apply the mask to the image
    segmented_image = image.copy()
    alpha_channel = np.zeros((H, W), dtype=np.uint8)
    alpha_channel[masks[1]] = 255
    rgba_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2RGBA)
    rgba_segmented_image[:, :, 3] = alpha_channel
    imageio.imwrite('segmented_image2.png', rgba_segmented_image)
    
    # Cut the image using the box coordinates
    segmented_image= cv2.imread("segmented_image2.png", cv2.IMREAD_UNCHANGED)
    cut_image = segmented_image[boxes_xyxy[1]:boxes_xyxy[3], boxes_xyxy[0]:boxes_xyxy[2]]
    imageio.imwrite('final_image.png', cut_image)
    
    #Embbeding image (CLIP)
    final_image= cv2.imread("final_image.png")
    inputs = processor(images=final_image, return_tensors="pt", padding=True).to(DEVICE)
    image_emb = model_clip.get_image_features(**inputs)
    image_emb = image_emb.squeeze(0).cpu().detach().numpy()
    
    #Querying Pinecone (the response varies depending on the searched namespace)
    pinecone.init(api_key="04d65f27-278f-40f0-9cfb-c907b84115a7", environment="us-east4-gcp")
    index = pinecone.Index("text-embeddings1")

    query_response = index.query(
        namespace="example-namespace",
        top_k=10,
        include_metadata=True,
        vector=image_emb.tolist())

    print(query_response.matches)
    filtered_ids = filter_top_scores(query_response.matches)
    
    return filtered_ids


@app.post("/save_embedding")
async def save_embedding(text_prompt: str, photo: UploadFile = File(...)):
    
    input_data = text_prompt
    image_bytes = io.BytesIO(await photo.read())
    image_np = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    
    
    output_filename = "original_image2.png"
    cv2.imwrite(output_filename, image)
    
    #Embbeding image (CLIP)
    original_image= cv2.imread("original_image2.png")
    inputs = processor(images=original_image, return_tensors="pt", padding=True).to(DEVICE)
    image_emb = model_clip.get_image_features(**inputs)
    image_emb = image_emb.squeeze(0).cpu().detach().numpy()
    
    #Saving embedding in Pinecone 
    pinecone.init(api_key="04d65f27-278f-40f0-9cfb-c907b84115a7", environment="us-east4-gcp")
    index_name = "text-embeddings1"
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension = 768, metric = "cosine")

    index = pinecone.Index(index_name)
    
    while True:
        vector_id = random.randint(1, 999999)
        fetch_response = index.fetch(ids=[str(vector_id)], namespace="example-namespace")
        if fetch_response["vectors"] == {}:
            break

    upsert_response = index.upsert(
        vectors=[(str(vector_id), image_emb.tolist(), {"element": input_data} )],
        namespace="example-namespace")

    print(index.describe_index_stats())

    
    
def filter_top_scores(input_list):
    if not input_list:
        return []

    # Find the element with the highest score
    highest_score_element = max(input_list, key=lambda x: x['score'])

    if highest_score_element["score"] < 0.5:
        answer = "Element not identified"
        return answer

    # Calculate the threshold (20% less than the highest score)
    threshold = highest_score_element['score'] * 0.95

    filtered_ids = []
    filtered_elements = []
    # Filter elements based on the threshold and return their 'id'
    for element in input_list:
        element_id = element['metadata']["ProductID"]
        element_name = element['metadata']["element"]
        if element['score'] >= threshold and element_id not in filtered_ids:
            filtered_elements.append((element_id, element_name))
            filtered_ids.append(element_id)
  

    return filtered_elements


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

