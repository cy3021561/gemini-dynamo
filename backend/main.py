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
    result = processor.retrieve_youtube_documents(str(request.youtube_link), verbose=True)
    
    # summary = gemini_processor.generate_documnet_summary(result, verbose=True)
    
    # Find key concepts
    raw_concepts = processor.find_key_concepts(result, verbose=True)
    
    # Deconstruct mutiple dicts
    unique_concepts = {}
    for concept_dict in raw_concepts:
        for key, value in concept_dict.items():
            unique_concepts[key] = value
    
    # Reconstruct to a list
    key_concepts_list = [{key: value} for key, value in unique_concepts.items()]
    
    return {
        "key_concepts": key_concepts_list
    }
