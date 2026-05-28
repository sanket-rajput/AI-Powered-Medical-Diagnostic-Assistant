import os
import requests

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

def ask_medical_bot(query):

    emergency_keywords = [
        "chest pain",
        "difficulty breathing",
        "shortness of breath",
        "stroke",
        "unconscious",
        "loss of consciousness"
    ]

    severity = "normal"

    for word in emergency_keywords:
        if word.lower() in query.lower():
            severity = "emergency"
            break

    docs = db.similarity_search(query, k=2)

    retrieved_context = [doc.page_content for doc in docs]
    context = "\n".join(retrieved_context)

    prompt = f"""
You are an AI-powered medical assistant.

Use ONLY the provided medical context.

Medical Context:
{context}

User Query:
{query}

Rules:
- Provide safe preliminary health guidance
- Do not diagnose diseases with certainty
- Mention possible concerns carefully
- Encourage professional consultation
- If symptoms appear severe or life-threatening,
  strongly recommend emergency medical attention

Respond professionally and clearly.
"""

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    data = response.json()

    return {
        "severity": severity,
        "response": data["choices"][0]["message"]["content"],
        "context": retrieved_context
    }
