from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import uvicorn

app = FastAPI()


# 1. Load the Model
DEVICE = torch.device("cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)

MODEL_PATH = "models/best_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# 2. Image Processing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 3. API Route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image sent form TypeScript
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    image_t = transform(image).unsqueeze(0).to(DEVICE)