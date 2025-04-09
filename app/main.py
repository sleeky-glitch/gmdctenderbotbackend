from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import pinecone
import httpx

from .models import ProjectDetails, TenderResponse
from .services import TenderGenerator
from .config import settings

app = FastAPI(
    title="Tender Generator API",
    description="API for generating tender documents using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    openai_client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        http_client=httpx.Client(
            timeout=60.0,
            follow_redirects=True
        )
    )

    # Initialize Pinecone
    pinecone.init(
        api_key=settings.PINECONE_API_KEY,
        environment="gcp-starter"  # replace with your environment
    )

    # Get the index
    pinecone_index = pinecone.Index(settings.PINECONE_INDEX_NAME)

    tender_generator = TenderGenerator(openai_client, pinecone_index)
except Exception as e:
    raise Exception(f"Failed to initialize services: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Tender Generator API"}

@app.post("/generate-tender", response_model=TenderResponse)
async def generate_tender(project_details: ProjectDetails):
    try:
        tender_sections = tender_generator.generate_complete_tender(
            project_details.model_dump()
        )
        return TenderResponse(sections=tender_sections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
