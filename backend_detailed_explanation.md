# HireGenei Backend: Comprehensive Technical Overview

## Project Overview
HireGenei is a web application that provides AI-powered job consulting services. The backend is built with FastAPI (Python) and provides three main functionalities:
- Resume analysis and skill extraction
- Job scraping from LinkedIn
- AI-powered chat consultation using Retrieval-Augmented Generation (RAG)

## Technology Stack
- **Framework**: FastAPI (modern, fast web framework for building APIs with Python)
- **Language**: Python 3.11.9
- **Key Libraries**:
  - `fastapi`: Web framework
  - `uvicorn`: ASGI server
  - `pydantic`: Data validation and serialization
  - `langchain`: LLM orchestration and RAG
  - `langchain-groq`: Groq API integration for LLM
  - `chromadb`: Vector database for embeddings
  - `sentence-transformers` (via HuggingFace): Text embeddings
  - `beautifulsoup4`: HTML parsing for web scraping
  - `pdfplumber`: PDF text extraction
  - `rapidfuzz`: Fuzzy string matching for skill extraction
- **External APIs**:
  - HuggingFace Inference API (for embeddings)
  - Groq API (for LLM chat)
- **Deployment**: Designed for cloud platforms like Render (supports PORT environment variable)

## Architecture Overview

### Directory Structure
```
backend/
├── main.py                 # Application entry point
├── core/
│   └── config.py          # Configuration and environment variables
├── models/
│   ├── request_models.py  # Pydantic request models
│   └── response_models.py # Pydantic response models
├── routes/
│   ├── chat.py           # Chat API endpoints
│   ├── jobs.py           # Job scraping endpoints
│   └── resume.py         # Resume upload endpoints
├── services/
│   ├── chat_service.py   # Chat business logic
│   ├── job_service.py    # Job scraping logic
│   └── resume_service.py # Resume processing logic
├── utils/
│   ├── embeddings.py     # HuggingFace embedding utilities
│   └── vectorstore.py    # Global vectorstore state management
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version specification
└── README.md             # Project documentation
```

### Application Flow
1. **Startup**: `main.py` initializes FastAPI app with CORS middleware and registers routers
2. **Request Handling**: Routes delegate to services for business logic
3. **Data Processing**: Services use utilities for embeddings and vector storage
4. **Response**: Structured responses using Pydantic models

## Core Components

### 1. Main Application (main.py)
- **Purpose**: Entry point and application configuration
- **Key Features**:
  - FastAPI app initialization
  - CORS configuration (allows all origins for development)
  - Router registration with prefixes (`/api/resume`, `/api/jobs`, `/api/chat`)
  - Health check endpoint (`/health`)
  - Root status endpoint (`/`)
  - Environment-aware port configuration (defaults to 8000, uses PORT env var for deployment)

### 2. Configuration (core/config.py)
- **Purpose**: Centralized configuration management
- **Features**:
  - Loads environment variables from `.env` file
  - Uses Pydantic Settings for validation
  - Manages API keys: `HF_TOKEN` (HuggingFace), `GROQ_API_KEY` (Groq)
  - Logs warnings if keys are missing

### 3. Data Models (models/)
- **Request Models** (`request_models.py`):
  - `JobScrapeRequest`: For job scraping (roles, cities, country, work_types, exp_levels, time_filter)
  - `ChatMessageRequest`: Simple message string for chat
- **Response Models** (`response_models.py`):
  - `ResumeUploadResponse`: Returns extracted skills and roles
  - `JobResponse`: Individual job details (title, company, location, link)
  - `JobScrapeResponse`: List of job responses
  - `ChatResponse`: AI chat reply

### 4. API Routes (routes/)

#### Resume Routes (resume.py)
- **Endpoint**: `POST /api/resume/upload`
- **Functionality**:
  - Accepts PDF file uploads only
  - Saves file temporarily
  - Extracts skills and roles using `resume_service.extract_resume_details()`
  - Triggers background task to build vectorstore
  - Returns skills and roles immediately (non-blocking)

#### Job Routes (jobs.py)
- **Endpoint**: `POST /api/jobs/scrape`
- **Functionality**:
  - Accepts job search criteria
  - Calls `job_service.scrape_jobs_async()` for LinkedIn scraping
  - Returns list of matching jobs
  - Uses async processing for non-blocking I/O

#### Chat Routes (chat.py)
- **Endpoint**: `POST /api/chat`
- **Functionality**:
  - Requires vectorstore to be initialized (resume uploaded first)
  - Uses RAG with resume context
  - Calls `chat_service.get_chat_response()`
  - Returns AI-generated response

### 5. Business Logic (services/)

#### Resume Service (resume_service.py)
- **Key Functions**:
  - `extract_resume_details(pdf_path)`: 
    - Uses pdfplumber to extract text
    - Fuzzy matches against predefined skills list (200+ skills)
    - Identifies job roles based on skill patterns
    - Returns skills list and roles list
  - `build_vectorstore_bg(pdf_path)`:
    - Background task to process resume
    - Uses LangChain PyPDFLoader and RecursiveCharacterTextSplitter
    - Creates embeddings via HuggingFace API
    - Stores in Chroma vectorstore for RAG

#### Job Service (job_service.py)
- **Key Function**: `scrape_jobs_async()`
- **Process**:
  - Constructs LinkedIn search URLs with filters
  - Uses requests with retry logic and user-agent rotation
  - Parses HTML with BeautifulSoup
  - Extracts job details (title, company, location, link)
  - Returns list of JobResponse objects
- **Features**:
  - Handles multiple roles, cities, work types, experience levels
  - Time-based filtering (24h, week, month)
  - Rate limiting and error handling

#### Chat Service (chat_service.py)
- **Key Function**: `get_chat_response(message)`
- **RAG Pipeline**:
  1. Retrieves vectorstore (resume chunks)
  2. Creates retriever from vectorstore
  3. Uses LangChain retrieval chain with Groq LLM
  4. Prompt includes user's skills/roles context
  5. Returns bullet-point responses
- **Dependencies**: Requires GROQ_API_KEY and initialized vectorstore

### 6. Utilities (utils/)

#### Embeddings (embeddings.py)
- **Purpose**: Generate text embeddings using HuggingFace
- **Key Function**: `get_embedding(text)`
- **Features**:
  - Uses `sentence-transformers/all-MiniLM-L6-v2` model
  - Handles rate limiting (503 responses) with exponential backoff
  - Supports single strings or lists of texts
  - Requires HF_TOKEN for API access

#### Vectorstore (vectorstore.py)
- **Purpose**: Global state management for vectorstore
- **Implementation**: Simple in-memory dictionary (MVP approach)
- **Functions**:
  - `set_vectorstore(vs)`: Store Chroma vectorstore
  - `get_vectorstore()`: Retrieve vectorstore
  - `set_skills/skills()`: Manage extracted skills
  - `set_roles/roles()`: Manage identified roles
- **Note**: In production, this would use Redis or database persistence

## Workflow Details

### Resume Upload Workflow
1. User uploads PDF resume
2. File validated (PDF only)
3. Temporary file created
4. Text extracted using pdfplumber
5. Skills extracted via fuzzy matching against SKILLS_LIST (200+ skills)
6. Roles identified based on skill combinations
7. Skills/roles returned immediately to frontend
8. Background task processes resume:
   - Loads PDF with LangChain PyPDFLoader
   - Splits text into chunks (RecursiveCharacterTextSplitter)
   - Generates embeddings for each chunk
   - Creates Chroma vectorstore
   - Stores globally for chat retrieval

### Job Scraping Workflow
1. User submits search criteria (roles, location, etc.)
2. Service constructs LinkedIn search URL with filters
3. Makes HTTP request with proper headers/user-agent
4. Parses HTML response with BeautifulSoup
5. Extracts job cards and details
6. Returns structured job list
7. Handles pagination and rate limiting

### Chat Consultation Workflow
1. User sends message
2. Router checks if vectorstore exists (resume uploaded)
3. If not, returns error
4. Retrieves relevant resume chunks using vectorstore.as_retriever()
5. Constructs prompt with user's skills/roles + retrieved context
6. Calls Groq API via LangChain
7. Returns AI response in bullet points

## Security and Configuration
- **Environment Variables**:
  - `HF_TOKEN`: HuggingFace API token
  - `GROQ_API_KEY`: Groq API key
  - `PORT`: Deployment port (optional)
- **CORS**: Configured for development (allows all origins)
- **File Handling**: Temporary files for PDF processing
- **API Keys**: Loaded from .env file or OS environment

## Deployment Considerations
- **Runtime**: Python 3.11.9 (specified in runtime.txt)
- **Server**: Uvicorn ASGI server
- **Port**: Configurable via PORT environment variable
- **Dependencies**: Listed in requirements.txt
- **Vectorstore**: Currently in-memory (would need persistence for production)

## Potential Improvements
- Replace global state with Redis/database
- Add authentication/authorization
- Implement rate limiting
- Add comprehensive error handling
- Use async file processing
- Add logging to external service
- Implement caching for embeddings
- Add API versioning
- Use Docker for containerization

## Interview Preparation Notes
- **Architecture**: Microservices-style with clear separation of concerns
- **Scalability**: Async processing, background tasks, external APIs
- **AI/ML Integration**: RAG pipeline, embeddings, vector search
- **Web Scraping**: Robust parsing, rate limiting, error handling
- **Data Processing**: PDF parsing, fuzzy matching, text chunking
- **API Design**: RESTful endpoints with Pydantic validation
- **Deployment**: Cloud-ready with environment configuration