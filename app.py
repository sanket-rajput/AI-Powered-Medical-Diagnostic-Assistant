from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import ask_medical_bot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def home():
    return {
        "message": "Medical RAG API Running"
    }

@app.post("/chat")
def chat(query: Query):

    result = ask_medical_bot(query.question)

    return {
        "question": query.question,
        "severity": result["severity"],
        "context": result["context"],
        "answer": result["response"]
    }
