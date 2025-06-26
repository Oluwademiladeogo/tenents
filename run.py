#!/usr/bin/env python3
"""
FastAPI startup script for the Tenets API
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    ) 