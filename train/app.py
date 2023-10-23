from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from transformers import AutoTokenizer
from fastapi.templating import Jinja2Templates
from typing import Optional
import torch
from vncorenlp import VnCoreNLP
from model import Combined_model
from utils import pred_to_label

'''
IMPORTANCE NOTE: Remember to cd to train/ before run the api 
'''

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory='templates')

# Load the trained model
MODEL_PATH = "weights/model_v3.pt"

# Initialize the model, tokenizer, and segmenter
model = Combined_model("vinai/phobert-base")
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
rdrsegmenter = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

# Create a Pydantic model for request body
class ReviewRequest(BaseModel):
    review_text: str

# Define an endpoint for model analysis
@app.post("/analyse")
async def analyse_review(review_data: ReviewRequest):
    review_sentence = review_data.review_text

    tokenized_input = tokenizer(review_sentence, return_tensors="pt", padding=True, truncation=True)
    tokenized_input.pop('token_type_ids', None)

    with torch.no_grad():
        outputs_classifier, outputs_regressor = model(**tokenized_input)
        outputs_classifier = outputs_classifier.cpu().numpy()
        outputs_regressor = outputs_regressor.cpu().numpy()
        outputs_regressor = outputs_regressor.argmax(axis=-1) + 1

        outputs = pred_to_label(outputs_classifier, outputs_regressor)

        return {
            "Giai_tri": int(outputs[0][0])*"⭐ ",
            "Luu_tru": int(outputs[0][1])*"⭐ ",
            "Nha_hang": int(outputs[0][2])*"⭐ ",
            "An_uong": int(outputs[0][3])*"⭐ ",
            "Di_chuyen": int(outputs[0][4])*"⭐ ",
            "Mua_sam": int(outputs[0][5])*"⭐ "
        }

@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
