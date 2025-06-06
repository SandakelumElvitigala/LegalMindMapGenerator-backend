from dotenv import load_dotenv
load_dotenv()

import os
import json
import uuid
import shutil
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import httpx
import asyncio
from pydantic import BaseModel
import tempfile
import docx2txt
import PyPDF2
from pathlib import Path

from utils import clean_and_parse_json, validate_graph_structure, sanitize_graph_data

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Legal Mind Map Generator")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage directory for uploaded files
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"

# Models
class LegalGraph(BaseModel):
    nodes: List[dict]
    edges: List[dict]

class GraphUpdateRequest(BaseModel):
    graph_id: str
    nodes: List[dict]
    edges: List[dict]

# In-memory storage for graphs (replace with database in production)
legal_graphs = {}

async def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF or DOCX files."""
    logger.info(f"Extracting text from file: {file_path}")
    file_path = Path(file_path)
    
    try:
        if file_path.suffix.lower() == '.pdf':
            logger.debug(f"Processing PDF file: {file_path}")
            text = ""
            with open(file_path, 'rb') as pdf_file:
                try:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    logger.debug(f"PDF has {len(pdf_reader.pages)} pages")
                    for page_num in range(len(pdf_reader.pages)):
                        logger.debug(f"Extracting text from page {page_num+1}")
                        text += pdf_reader.pages[page_num].extract_text() + "\n"
                    logger.debug(f"Successfully extracted {len(text)} characters from PDF")
                except Exception as e:
                    logger.error(f"Error reading PDF: {str(e)}", exc_info=True)
                    raise ValueError(f"Error reading PDF: {str(e)}")
            return text
        
        elif file_path.suffix.lower() == '.docx':
            logger.debug(f"Processing DOCX file: {file_path}")
            try:
                text = docx2txt.process(file_path)
                logger.debug(f"Successfully extracted {len(text)} characters from DOCX")
                return text
            except Exception as e:
                logger.error(f"Error reading DOCX: {str(e)}", exc_info=True)
                raise ValueError(f"Error reading DOCX: {str(e)}")
        
        else:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    except Exception as e:
        logger.error(f"Error in extract_text_from_file: {str(e)}", exc_info=True)
        raise


async def analyze_legal_text(text: str) -> dict:
    """Send text to Groq API for analysis and entity extraction."""
    logger.info("Starting legal text analysis")
    
    if not GROQ_API_KEY:
        logger.error("Groq API key not configured")
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    
    # Truncate text if too long (adjust based on model limits)
    max_text_length = 15000
    original_length = len(text)
    logger.debug(f"Text length: {original_length} characters")
    
    if original_length > max_text_length:
        text = text[:max_text_length] + "...[truncated]"
        logger.info(f"Text truncated from {original_length} to {max_text_length} characters")
    
    system_prompt = """
        You are a legal document analyzer. Extract key information from the legal document and organize it into a structured graph.
        Identify and extract:
        1. Entities (people, organizations, courts, laws, etc.)
        2. Incidents (events, crimes, hearings, filings, etc.)
        3. Clues (evidence, testimonies, documents, forensic findings, etc.)
        4. Places (locations, venues, jurisdictions, etc.)
        5. Relationships between these items
        6. All dates, case numbers, amounts, and specific details

        For each entity, incident, or clue, be sure to include any associated location information.

        Return ONLY a valid JSON object with this structure:
        {
        "nodes": [
            {
            "id": "unique_id", 
            "type": "Entity|Incident|Clue|Place", 
            "label": "Short Name", 
            "details": "Additional details",
            "place": "Location name if applicable",
            "date" : "Date if applicable",
            "amount": "Amount if applicable",
            "actions": ["List of actions"],
            "case_references": ["References"]
            }
        ],
        "edges": [
            {
            "id": "unique_id", 
            "source": "source_node_id", 
            "target": "target_node_id", 
            "label": "Relationship Type"
            }
        ]
        }
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this legal document and extract entities, incidents, and clues as a graph:\n\n{text}"}
    ]
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.debug("Preparing to send request to Groq API")
        
        request_body = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4000
        }
        
        logger.debug(f"Using model: mixtral-8x7b-32768")
        
        try:
            async with httpx.AsyncClient() as client:
                logger.debug(f"Sending request to {GROQ_API_URL}")
                
                response = await client.post(
                    GROQ_API_URL,
                    headers=headers,
                    json=request_body,
                    timeout=60.0
                )
                
                logger.debug(f"Received response with status code: {response.status_code}")
                
                if response.status_code != 200:
                    response_text = response.text
                    logger.error(f"Groq API error: Status {response.status_code}, Response: {response_text}")
                    raise HTTPException(status_code=response.status_code, 
                                      detail=f"Groq API error: {response_text}")
                
                result = response.json()
                logger.debug("Successfully parsed JSON response from API")
                
                # Extract the JSON from the response
                json_text = result["choices"][0]["message"]["content"]
                logger.debug(f"Raw content from API: {json_text[:100]}...")
                
                # Find the JSON part if there's extra text
                start_idx = json_text.find('{')
                end_idx = json_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    logger.debug(f"Extracted JSON content from position {start_idx} to {end_idx}")
                    json_text = json_text[start_idx:end_idx]
                else:
                    logger.warning("Could not find JSON boundaries in response")
                
                # Parse the JSON
                try:
                    logger.debug("Attempting to parse JSON from response")
                    graph_data = clean_and_parse_json(json_text)
                    
                    # Check if there was a parsing error
                    if "error" in graph_data:
                        logger.error(f"JSON parsing resulted in error: {graph_data['error']}")
                        # Return a basic structure instead of failing completely
                        graph_data = {"nodes": [], "edges": []}
                    
                    # Validate and sanitize the structure
                    if not validate_graph_structure(graph_data):
                        logger.error("Invalid graph structure received")
                        raise ValueError("Invalid graph structure received")
                    
                    # Clean up the data
                    graph_data = sanitize_graph_data(graph_data)
                    
                    logger.info(f"Successfully extracted graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
                    return graph_data
                    
                except Exception as parse_error:
                    logger.error(f"Failed to parse JSON after cleanup: {parse_error}", exc_info=True)
                    logger.error(f"Final JSON text: {json_text}")
                    raise HTTPException(status_code=422, 
                                    detail=f"Failed to parse JSON from Groq API response: {str(parse_error)}")
                
        except httpx.TimeoutException:
            logger.error("Request to Groq API timed out")
            raise HTTPException(status_code=504, detail="Request to Groq API timed out")
        except httpx.HTTPError as e:
            logger.error(f"HTTP error with Groq API: {str(e)}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"HTTP error with Groq API: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in analyze_legal_text: {str(e)}", exc_info=True)

@app.post("/upload/", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """Upload a legal document and process it."""
    logger.info(f"Received upload request for file: {file.filename}")
    
    try:
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        logger.debug(f"File extension: {file_ext}")
        
        if file_ext not in [".pdf", ".docx"]:
            logger.warning(f"Invalid file extension: {file_ext}")
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")
        
        # Generate a unique ID for this analysis
        graph_id = str(uuid.uuid4())
        logger.debug(f"Generated graph ID: {graph_id}")
        
        # Create directory for this upload if it doesn't exist
        upload_path = UPLOAD_DIR / graph_id
        upload_path.mkdir(exist_ok=True)
        logger.debug(f"Created upload directory at: {upload_path}")
        
        # Save the uploaded file
        file_path = upload_path / file.filename
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        logger.info(f"File saved at: {file_path}")
        
        # Extract text
        extracted_text = await extract_text_from_file(str(file_path))
        logger.info("Text extraction complete")
        
        # Analyze text
        graph_data = await analyze_legal_text(extracted_text)
        logger.info("Text analysis complete")
        
        # Save graph in memory (could be replaced with persistent storage)
        legal_graphs[graph_id] = graph_data
        
        return {
            "graph_id": graph_id,
            "graph": graph_data
        }
    
    except Exception as e:
        logger.error(f"Unhandled error in upload_document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/graph/{graph_id}", response_model=dict)
async def get_graph(graph_id: str):
    """Retrieve a stored graph by ID."""
    logger.info(f"Fetching graph with ID: {graph_id}")
    
    if graph_id not in legal_graphs:
        logger.warning(f"Graph not found: {graph_id}")
        raise HTTPException(status_code=404, detail="Graph not found")
    
    logger.info(f"Returning graph with {len(legal_graphs[graph_id]['nodes'])} nodes and {len(legal_graphs[graph_id]['edges'])} edges")
    return {
        "graph_id": graph_id,
        "graph": legal_graphs[graph_id]
    }

@app.put("/graph/{graph_id}", response_model=dict)
async def update_graph(graph_id: str, graph_update: GraphUpdateRequest):
    """Update a stored graph."""
    logger.info(f"Updating graph with ID: {graph_id}")
    
    if graph_id not in legal_graphs:
        logger.warning(f"Graph not found: {graph_id}")
        raise HTTPException(status_code=404, detail="Graph not found")
    
    # Update the stored graph
    legal_graphs[graph_id] = {
        "nodes": graph_update.nodes,
        "edges": graph_update.edges
    }
    
    logger.info(f"Graph updated successfully")
    return {
        "graph_id": graph_id,
        "graph": legal_graphs[graph_id]
    }

@app.delete("/graph/{graph_id}")
async def delete_graph(graph_id: str):
    """Delete a stored graph."""
    logger.info(f"Deleting graph with ID: {graph_id}")
    
    if graph_id not in legal_graphs:
        logger.warning(f"Graph not found: {graph_id}")
        raise HTTPException(status_code=404, detail="Graph not found")
    
    # Remove the graph data
    del legal_graphs[graph_id]
    
    # Remove associated files
    graph_dir = UPLOAD_DIR / graph_id
    if graph_dir.exists():
        try:
            shutil.rmtree(graph_dir)
            logger.info(f"Removed directory: {graph_dir}")
        except Exception as e:
            logger.error(f"Failed to remove directory: {str(e)}", exc_info=True)
    
    return {"status": "deleted"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)