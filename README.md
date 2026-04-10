# HireGenei Backend

This folder contains the FastAPI backend for the HireGenei application.

## Overview

The backend exposes API endpoints for resume analysis, job scraping, and AI chat functionality.

## Structure

- `main.py` - FastAPI application entrypoint with CORS configuration and router registration.
- `routes/` - API route definitions:
  - `resume.py` - resume upload and analysis endpoints
  - `jobs.py` - job scraping endpoints
  - `chat.py` - AI chat endpoints
- `services/` - business logic used by routes.
- `models/` - request/response Pydantic models.
- `core/config.py` - central configuration utilities.
- `utils/` - helper utilities for embeddings and vector storage.
- `.env` - environment variables for local development.

## Requirements

This backend uses Python and the libraries listed in `requirements.txt`.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app locally:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application also supports the `PORT` environment variable when running in hosted environments.

## API Endpoints

- `GET /health` - health check endpoint
- `GET /` - root status endpoint
- `POST /api/resume/upload` - resume upload and analysis
- `POST /api/jobs/scrape` - scrape jobs based on request criteria
- `POST /api/chat` - AI job consultant chat endpoint

## Notes

- CORS is configured to allow all origins for development.
- The backend uses FastAPI, Pydantic, LangChain, sentence-transformers, FAISS, and other dependencies for AI/processing workflows.
