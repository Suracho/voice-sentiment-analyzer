from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
import shutil
import os
import ml
from pathlib import Path
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
    name="static",
)

@app.get("/")
def home(request : Request):
    return templates.TemplateResponse("index.html", {"request" : request})

@app.get("/upload")
def peep(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/analysis")
def perform_analysis(audio: UploadFile = File(...)):
    temp_file = _save_file_to_disk(audio,path="temp", save_as="temp")
    text = ml.sentiment_analysis(temp_file)
    return {"filename": audio.filename, "text" : text}

def _save_file_to_disk(uploaded_file, path=".", save_as="default"):
    extension = os.path.splitext(uploaded_file.filename)[-1]
    temp_file = os.path.join(path, save_as + extension)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file