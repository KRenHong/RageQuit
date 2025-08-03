from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from classify import classify_audio
import tempfile
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    label, top5 = classify_audio(tmp_path)
    return {
        "label": label,
        "top_5": top5
    }
