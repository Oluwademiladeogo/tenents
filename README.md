---
title: Tenets
emoji: 🐠
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
short_description: service for tenets rankings
---

# Tenets API - FastAPI Version

A FastAPI-based API for phonetic analysis and rankings, converted from the original Flask application.

## Features

- **FastAPI**: Modern, fast web framework with automatic API documentation
- **Async Support**: Better performance with async/await patterns
- **Type Safety**: Pydantic models for request/response validation
- **Auto Documentation**: Interactive API docs at `/docs` and `/redoc`
- **CORS Support**: Cross-origin resource sharing enabled

## API Endpoints

### POST `/save-rankings`

Processes phonetic rankings and returns analysis results.

**Request Body:**

```json
{
  "word": "example",
  "ipa_variants": [...],
  "confusion_matrix": {...}
}
```

**Response:**

```json
{
  "targetWord": "example",
  "bestTranscription": "/ɪɡˈzæmpəl/",
  "finalTable": {...}
}
```

### GET `/`

Health check endpoint.

### GET `/health`

Health status endpoint.

## API Documentation

Once the server is running, you can access:

- **Interactive API docs**: http://localhost:5000/docs
- **ReDoc documentation**: http://localhost:5000/redoc

## Key Changes from Flask Version

1. **Framework**: Flask → FastAPI
2. **Async Support**: Added async/await for better performance
3. **Type Safety**: Pydantic models for request/response validation
4. **Error Handling**: HTTPException instead of Flask error responses
5. **Documentation**: Automatic OpenAPI/Swagger documentation
6. **CORS**: FastAPI middleware instead of Flask-CORS

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
