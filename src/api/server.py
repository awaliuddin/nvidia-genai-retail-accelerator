#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API server for the Retail GenAI system.

This module provides a FastAPI server that exposes the functionality
of the Retail GenAI system through RESTful endpoints.
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
from io import BytesIO
import base64

# Ensure the repository root is in the Python path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Import project modules
from src.models.multimodal_fusion import RetailProductFusionModel, create_nvidia_optimized_fusion_model
from src.inference.pipeline import RetailGenAIPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("retail_genai_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
MODEL_DIR = os.environ.get("MODEL_DIR", str(repo_root / "models"))
USE_GPU = os.environ.get("USE_GPU", "True").lower() in ("true", "1", "t")

# Define API models
class ProductQuery(BaseModel):
    image_base64: str
    text: str

class ProductQuestion(BaseModel):
    image_base64: str
    question: str

class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    models_loaded: bool

# Initialize the API
app = FastAPI(
    title="Retail GenAI API",
    description="Multi-modal API for retail applications powered by NVIDIA GPUs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and pipeline
pipeline = None

# Load models on startup
@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        logger.info("Initializing models...")
        device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
        
        # Initialize the pipeline with models
        pipeline = RetailGenAIPipeline.from_pretrained(
            model_dir=MODEL_DIR,
            device=device
        )
        
        logger.info(f"Models initialized successfully. Using device: {device}")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_available = torch.cuda.is_available() and USE_GPU
    models_loaded = pipeline is not None
    
    if not models_loaded:
        status = "warning"
    else:
        status = "ok"
    
    return {
        "status": status,
        "version": "1.0.0",
        "gpu_available": gpu_available,
        "models_loaded": models_loaded
    }

# Product classification endpoint
@app.post("/predict")
async def predict_product(query: ProductQuery):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(query.image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Run prediction
        start_time = time.time()
        result = pipeline.predict(image, query.text)
        processing_time = time.time() - start_time
        
        # Add processing time
        result["processing_time"] = processing_time
        
        return result
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Product Q&A endpoint
@app.post("/answer")
async def answer_question(query: ProductQuestion):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(query.image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Answer question
        start_time = time.time()
        result = pipeline.answer_product_question(image, query.question)
        processing_time = time.time() - start_time
        
        # Add processing time
        result["processing_time"] = processing_time
        
        return result
    except Exception as e:
        logger.error(f"Error in answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Shelf analysis endpoint
@app.post("/analyze_shelf")
async def analyze_shelf(image: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Process shelf image
        start_time = time.time()
        result = pipeline.process_shelf_image(pil_image)
        processing_time = time.time() - start_time
        
        # Add processing time
        result["processing_time"] = processing_time
        
        # Create visualization
        viz_image = pipeline.visualize_shelf_detection(pil_image, result)
        
        # Convert to base64 for response
        buffered = BytesIO()
        viz_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Add visualization to result
        result["visualization_base64"] = img_str
        
        return result
    except Exception as e:
        logger.error(f"Error in shelf analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main function to run the API server
def main():
    logger.info(f"Starting Retail GenAI API server on {HOST}:{PORT}")
    uvicorn.run("src.api.server:app", host=HOST, port=PORT, reload=DEBUG)

if __name__ == "__main__":
    main()
