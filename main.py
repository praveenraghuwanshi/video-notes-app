import os
from fastapi import FastAPI, HTTPException
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
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "live/" in url:
        return url.split("live/")[1].split("?")[0]
    return url

@app.get("/")
def home():
    return {"status": "backend operational"}

@app.post("/api/convert")
async def convert_video(req: VideoRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Railway variables.")

    client = genai.Client(api_key=api_key)

    try:
        # 1. Extract Video ID & Fetch Transcript Directly
        video_id = extract_video_id(req.url)
        
        # Try fetching transcript in Hindi, English, or auto-generated
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi', 'en-IN'])
        transcript_text = " ".join([t['text'] for t in transcript_list])

        # 2. Prompt Gemini AI
        prompt = f"""
        Analyze this video transcript and generate structured study notes in language: {req.language}.
        Output valid JSON ONLY matching this exact structure:
        {{
          "title": "Main Title",
          "sec1Title": "1. Section Heading",
          "sec1Body": "Detailed text summary of key concepts explained in the video",
          "points": ["Key Point 1", "Key Point 2", "Key Point 3"],
          "aiInsight": "AI Extra Context or Smart Tip",
          "diagramLabels": ["Label 1", "Label 2", "Label 3"]
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
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcript Error: {str(e)}")
        
