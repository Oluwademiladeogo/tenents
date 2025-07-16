from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os
import logging
from datetime import datetime
from phonetic_analyzer import PhoneticAnalyzer, AnalysisRequest
from config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=Config.CORS_CREDENTIALS,
    allow_methods=Config.CORS_METHODS,
    allow_headers=Config.CORS_HEADERS,
)

# Initialize the phonetic analyzer
analyzer = PhoneticAnalyzer()

# Pydantic models for request/response
class RankingsRequest(BaseModel):
    word: str
    ipa_variants: List[Dict[str, Any]]
    confusion_matrix: Dict[str, Any]
    sliders: Dict[str, int]  

class RankingsResponse(BaseModel):
    targetWord: str
    bestTranscription: str
    finalTable: Dict[str, Dict[str, float]]

class ErrorResponse(BaseModel):
    error: str

@app.post('/save-rankings', response_model=RankingsResponse)
async def save_rankings(request: RankingsRequest):
    try:
        # Validate request data
        if not request.word or not request.ipa_variants or not request.confusion_matrix:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        if not request.sliders:
            raise HTTPException(status_code=400, detail="Slider preferences are required")
        
        logger.info(f"Processing analysis for word: {request.word}")
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            word=request.word,
            ipa_variants=request.ipa_variants,
            confusion_matrix=request.confusion_matrix,
            sliders=request.sliders
        )
        
        result = analyzer.analyze(analysis_request)
        
        logger.info(f"Analysis completed. Best transcription: {result.best_transcription}")
        
        # Build response
        response_data = RankingsResponse(
            targetWord=result.target_word,
            bestTranscription=result.best_transcription,
            finalTable=result.final_table
        )
        
        return response_data
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {"error": "Internal server error", "detail": str(exc)}

@app.get("/")
async def root():
    return {"message": "Tenets API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app, 
        host=Config.HOST, 
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower()
    )