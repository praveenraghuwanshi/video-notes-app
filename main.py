import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
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

def extract_video_id(url: str) -> str:
    patterns = [
        r'(?:v=|\/live\/|\/v\/|youtu\.be\/|\/embed\/)([^"&?\/\s]{11})',
        r'^([^"&?\/\s]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url.strip()

@app.get("/")
def home():
    return {"status": "backend operational"}

@app.post("/api/convert")
async def convert_video(req: VideoRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"status": "error", "message": "GEMINI_API_KEY missing in Railway variables."}

    try:
        video_id = extract_video_id(req.url)

        # 1. Fetch transcript from YouTube directly
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['hi', 'en', 'hi-IN', 'en-IN'])
        except Exception:
            transcript = transcript_list.find_transcript([])

        fetched_data = transcript.fetch()
        transcript_text = " ".join([t['text'] for t in fetched_data])

        if not transcript_text.strip():
            return {"status": "error", "message": "Could not extract transcript from video."}

        # 2. Pass real transcript to Gemini
        client = genai.Client(api_key=api_key)
        prompt = f"""
        Analyze this video transcript and generate detailed, accurate study notes in language: {req.language}.
        Output valid JSON ONLY matching this exact structure:
        {{
          "title": "Main Title summarizing video subject",
          "sec1Title": "1. Key Topic Heading",
          "sec1Body": "Detailed summary explaining the exact facts and announcements made in the transcript.",
          "points": ["Key Point 1", "Key Point 2", "Key Point 3"],
          "aiInsight": "Smart AI Tip or Summary based on the transcript",
          "diagramLabels": ["Concept 1", "Process 2", "Outcome 3"]
        }}
        Transcript: {transcript_text[:15000]}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )

        return {"status": "success", "data": response.text}

    except Exception as e:
        print(f"BACKEND ERROR: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}
        
