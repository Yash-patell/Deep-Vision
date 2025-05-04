
from fastapi import FastAPI, File, UploadFile, WebSocket, Request
import cv2
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
torch.backends.cudnn.benchmark = True
import torchvision.transforms as transforms
from PIL import Image
import timm
import shutil
import os
import torch.nn as nn
app = FastAPI()
from fastapi.staticfiles import StaticFiles
import asyncio

# Serve the temp_videos directory so the video can be accessed by the frontend
app.mount("/temp_videos", StaticFiles(directory="temp_videos"), name="temp_videos")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained Xception model
model = timm.create_model('legacy_xception', pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('./trained_models/xception_fine_tune_25-frames.pth', map_location=device))
model.eval()
model.to(device)

# Transform for frames
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Set up templates for serving frontend
templates = Jinja2Templates(directory="templates")

# Route to serve the HTML frontend
@app.get("/home", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# Upload video endpoint
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    video_path = f"temp_videos/{file.filename}"
    os.makedirs("temp_videos", exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}

# WebSocket for live prediction
@app.websocket("/live-predict/")
async def live_predict(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WebSocket connection accepted")

    try:
        video_data = await websocket.receive_json()
        print(f"🟡 Received data: {video_data}")

        video_path = f"temp_videos/{video_data['filename']}"
        print(f"🟠 Looking for video at: {video_path}")

        if not os.path.exists(video_path):
            print(f"🔴 File not found: {video_path}")
            await websocket.send_json({"error": f"File not found: {video_path}"})
            await websocket.close()
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"🔴 Failed to open video: {video_path}")
            await websocket.send_json({"error": "Could not open video"})
            await websocket.close()
            return

        print("✅ Video opened successfully")

        with torch.no_grad():
            while True:  # Changed to infinite loop, controlled by client
                video_data = await websocket.receive_json()  # Receive timestamp
                timestamp = video_data.get('timestamp')
                if timestamp is None:
                    print("🛑 No timestamp received, closing connection")
                    break

                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Set video position

                ret, frame = cap.read()
                if not ret:
                    print("🛑 End of video or error reading frame")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

                output = model(frame_tensor)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()

                result = {
                    "timestamp": timestamp,  # Include timestamp in result
                    "label": "REAL" if prediction == 0 else "FAKE",
                    "confidence": round(confidence * 100, 2)
                }

                await websocket.send_json(result)

        cap.release()
        os.remove(video_path)
        print(f"🧹 Cleaned up video: {video_path}")
        await websocket.close()

    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        await websocket.send_json({"error": str(e)})
        await websocket.close()