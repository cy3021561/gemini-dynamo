from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from services.genai import YoutubeProcessor, GeminiProcessor

# Define the request class
class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl

app = FastAPI()

# Configure CORS: allow request from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_processor = GeminiProcessor(
        model_name = "gemini-pro",
        project = "gemini-v1-423923"
        )

@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):
    # Analizing
    processor = YoutubeProcessor(gemini_processor = gemini_processor)
    result = processor.retrieve_youtube_documents(str(request.youtube_link))
    
    # summary = gemini_processor.generate_documnet_summary(result, verbose=True)
    # Find key concepts
    key_concepts = processor.find_key_concepts(result, group_amount=2)
    
    return {
        "key_concepts": key_concepts
    }
