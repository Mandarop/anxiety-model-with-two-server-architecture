from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()

# Load MuRIL once
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
muril_model = AutoModel.from_pretrained("google/muril-base-cased")
muril_model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/embed/")
async def get_embedding(req: TextRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        embedding = muril_model(**inputs).last_hidden_state[:, 0, :]
    return {"embedding": embedding.squeeze(0).tolist()}
