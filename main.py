import os
import re
from fastapi import FastAPI
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

def normalize_youtube_url(url: str) -> str:
    patterns = [
        r'(?:v=|\/live\/|\/v\/|youtu\.be\/|\/embed\/)([^"&?\/\s]{11})',
        r'^([^"&?\/\s]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
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
        client = genai.Client(api_key=api_key)
        clean_url = normalize_youtube_url(req.url)

        # YouTube URI requires explicit mime_type="video/mp4"
        video_part = types.Part(
            file_data=types.FileData(
                file_uri=clean_url,
                mime_type="video/mp4"
            )
        )

        prompt_part = types.Part(text=f"""
        Analyze this YouTube video and generate structured study notes in language: {req.language}.
        Output valid JSON ONLY matching this exact structure:
        {{
          "title": "Main Title of Video",
          "sec1Title": "1. Main Topic Heading",
          "sec1Body": "Detailed summary explaining core concepts presented in the video.",
          "points": ["Key Takeaway 1", "Key Takeaway 2", "Key Takeaway 3"],
          "aiInsight": "Smart AI Insight or Exam Tip based on this video",
          "diagramLabels": ["Concept 1", "Process 2", "Outcome 3"]
        }}
        """)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=types.Content(parts=[video_part, prompt_part]),
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )

        return {"status": "success", "data": response.text}

    except Exception as e:
        print(f"BACKEND ERROR: {str(e)}")
        return {"status": "error", "message": f"Gemini Error: {str(e)}"}
        
