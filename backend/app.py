from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipelines import run_pipeline
from src.corpus import build_index

app = FastAPI()

# Enable CORS for React frontend (usually on port 3000 or 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    pipeline: str
    top_k: int = 5

@app.on_event("startup")
async def startup_event():
    # Ensure index is built on startup (checks if exists)
    print("Initializing index...")
    build_index(limit=None) # Full build if not already cached

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        print(f"Processing query: {request.query} using {request.pipeline}")
        answer, context = run_pipeline(request.pipeline, request.query, top_k=request.top_k)
        
        # Format context for frontend
        formatted_context = []
        for c in context:
            formatted_context.append({
                "text": c["text"],
                "score": float(c["score"]),
                "metadata": c["metadata"]
            })
            
        return {
            "query": request.query,
            "pipeline": request.pipeline,
            "answer": answer,
            "context": formatted_context
        }
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
