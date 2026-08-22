import os
import tempfile
import yt_dlp
import whisper
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
whisper_model = whisper.load_model("base")

class VideoRequest(BaseModel):
    url: str
    language: str = "en"

def download_audio(video_url: str, output_dir: str) -> str:
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': output_template,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        return os.path.splitext(ydl.prepare_filename(info))[0] + ".mp3"

@app.post("/api/convert")
async def convert_video(req: VideoRequest):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            audio_path = download_audio(req.url, temp_dir)
            transcript = whisper_model.transcribe(audio_path).get("text", "")

            prompt = f"""
            Analyze this video transcript and generate structured study notes in {req.language}.
            Output JSON matching this exact structure:
            {{
              "title": "Main Title",
              "sec1Title": "1. Section Heading",
              "sec1Body": "Detailed text summary",
              "points": ["Point 1", "Point 2"],
              "aiInsight": "AI Extra Insight",
              "diagramLabels": ["Label 1", "Label 2", "Label 3"]
            }}
            Transcript: {transcript}
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return {"status": "success", "data": response.text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
