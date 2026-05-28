import os
import requests

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class DummyEmbeddings:
    def embed_query(self, text):
        return [0.0] * 384

    def embed_documents(self, texts):
        return [[0.0] * 384 for _ in texts]


embeddings = DummyEmbeddings()

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

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI medical assistant.

Use ONLY the provided context.

Medical Context:
{context}

User Query:
{query}

Rules:
- Give preliminary guidance
- Do not diagnose with certainty
- Recommend professional consultation
- Mention emergency care if severe
"""

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek/deepseek-r1:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        timeout=60
    )

    data = response.json()

    print("OPENROUTER RESPONSE:", data)

    if "choices" not in data:
        return {
            "severity": severity,
            "response": f"OpenRouter API Error: {data}"
        }

    return {
        "severity": severity,
        "response": data["choices"][0]["message"]["content"]
    }
