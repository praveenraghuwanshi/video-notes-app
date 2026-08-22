import os
import tempfile
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str
    language: str = "en"

def download_audio(video_url: str, output_dir: str) -> str:
    output_template = os.path.join(output_dir, "audio.%(ext)s")
    ydl_opts = {
        'format': 'ba/ba*',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128'}],
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(video_url, download=True)
        return os.path.join(output_dir, "audio.mp3")

@app.get("/")
def home():
    return {"status": "backend running"}

@app.post("/api/convert")
async def convert_video(req: VideoRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in Railway Variables.")

    client = genai.Client(api_key=api_key)

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 1. Fast Audio Download
            audio_path = download_audio(req.url, temp_dir)

            # 2. Upload Audio File Directly to Gemini AI
            audio_file = client.files.upload(file=audio_path)

            # 3. Request Transcription + Notes in One Shot
            prompt = f"""
            Listen to this video audio and generate structured study notes in language: {req.language}.
            Output valid JSON matching this exact structure:
            {{
              "title": "Main Title",
              "sec1Title": "1. Section Heading",
              "sec1Body": "Detailed summary of key concepts explained in the video",
              "points": ["Key Point 1", "Key Point 2", "Key Point 3"],
              "aiInsight": "AI Extra Context or Smart Tip",
              "diagramLabels": ["Label 1", "Label 2", "Label 3"]
            }}
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[audio_file, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )

            # Clean up audio from Gemini Storage
            client.files.delete(name=audio_file.name)

            return {"status": "success", "data": response.text}

        except Exception as e:
            print(f"PIPELINE ERROR: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
