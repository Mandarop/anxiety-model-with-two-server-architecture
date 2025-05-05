from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI()

# Symptom labels
anxiety_labels = [
    "Symptom 10", "Symptom 11", "Symptom 12",
    "Symptom 13", "Symptom 14", "Symptom 15", "Symptom 16"
]

# Model class
class MultiLabelNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# Request schema
class EmbeddingRequest(BaseModel):
    embedding: list

@app.post("/predict/")
async def predict(req: EmbeddingRequest):
    model = MultiLabelNet(input_dim=768, output_dim=7)
    model.load_state_dict(torch.load("anxiety.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        input_tensor = torch.tensor(req.embedding).unsqueeze(0)
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0)

    threshold = 0.3
    predicted = [lab for lab, keep in zip(anxiety_labels, (probs > threshold).tolist()) if keep]

    del model
    torch.cuda.empty_cache()

    return {"predicted_symptoms": predicted}
