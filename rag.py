import logging
import os
from threading import Lock

import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# =========================
# Logging Configuration
# =========================

logger = logging.getLogger("rag")

logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)

# =========================
# Lightweight Embeddings
# =========================

class DummyEmbeddings(Embeddings):

    def embed_query(self, text):
        return [0.0] * 384

    def embed_documents(self, texts):
        return [[0.0] * 384 for _ in texts]

# =========================
# Load FAISS Vector Store
# =========================

EMBEDDINGS = DummyEmbeddings()

DB = FAISS.load_local(
    "vectorstore",
    EMBEDDINGS,
    allow_dangerous_deserialization=True
)

# =========================
# Load OpenRouter Keys
# =========================

def load_openrouter_keys():

    keys = []

    for i in range(1, 15):

        key = os.getenv(f"OPENROUTER_API_KEY_{i}")

        if key:
            keys.append(key)

    fallback_key = os.getenv("OPENROUTER_API_KEY")

    if fallback_key and fallback_key not in keys:
        keys.insert(0, fallback_key)

    return keys

OPENROUTER_KEYS = load_openrouter_keys()

KEY_COUNT = len(OPENROUTER_KEYS)

ACTIVE_KEY_INDEX = 0

KEY_LOCK = Lock()

# =========================
# Stable Working Models
# =========================

MODEL_CANDIDATES = [
    "openrouter/free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-7b-instruct:free"
]

# =========================
# Retryable Status Codes
# =========================

def is_retryable_status(status_code):

    return status_code in {
        401,
        403,
        404,
        408,
        409,
        422,
        429,
        500,
        502,
        503,
        504
    }

# =========================
# OpenRouter API Caller
# =========================

def call_openrouter(prompt, timeout=60):

    global ACTIVE_KEY_INDEX

    if KEY_COUNT == 0:

        logger.error("No OpenRouter API keys configured.")

        return {
            "success": False,
            "error": "No OpenRouter API keys configured"
        }

    attempts = 0

    failed_keys = []

    for model in MODEL_CANDIDATES:

        logger.info(f"Trying model: {model}")

        for offset in range(KEY_COUNT):

            with KEY_LOCK:
                key_index = (ACTIVE_KEY_INDEX + offset) % KEY_COUNT

            api_key = OPENROUTER_KEYS[key_index]

            attempts += 1

            logger.info(
                f"Attempt {attempts}: model={model} key_index={key_index}"
            )

            try:

                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    },
                    timeout=timeout
                )

                # =========================
                # Handle HTTP Errors
                # =========================

                try:
                    response.raise_for_status()

                except requests.HTTPError:

                    status_code = response.status_code

                    logger.warning(
                        f"Key {key_index} failed "
                        f"for model {model} "
                        f"with HTTP {status_code}: "
                        f"{response.text}"
                    )

                    failed_keys.append({
                        "key_index": key_index,
                        "model": model,
                        "status": status_code
                    })

                    # Model unavailable
                    if status_code == 404:
                        break

                    if is_retryable_status(status_code):
                        continue

                    continue

                # =========================
                # Parse JSON
                # =========================

                try:
                    data = response.json()

                except ValueError:

                    logger.warning(
                        f"Non JSON response from "
                        f"key {key_index}"
                    )

                    continue

                logger.info(
                    f"SUCCESS model={model} "
                    f"key_index={key_index}"
                )

                # =========================
                # Validate Response
                # =========================

                if "choices" not in data:

                    logger.warning(
                        f"Malformed response: {data}"
                    )

                    continue

                # =========================
                # Update Active Key
                # =========================

                with KEY_LOCK:
                    ACTIVE_KEY_INDEX = key_index

                return {
                    "success": True,
                    "data": data,
                    "used_key_index": key_index,
                    "used_model": model,
                    "attempts": attempts
                }

            except requests.Timeout:

                logger.warning(
                    f"Timeout for key {key_index}"
                )

                continue

            except requests.RequestException as exc:

                logger.warning(
                    f"Request failed for key "
                    f"{key_index}: {exc}"
                )

                continue

    logger.error(
        f"All OpenRouter attempts failed "
        f"after {attempts} attempts"
    )

    return {
        "success": False,
        "error": "All OpenRouter keys/models failed",
        "failed_keys": failed_keys
    }

# =========================
# Main Medical Bot
# =========================

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

    # =========================
    # Vector Retrieval
    # =========================

    try:

        docs = DB.similarity_search(query, k=2)

    except Exception as exc:

        logger.exception("Vector retrieval failed")

        return {
            "severity": severity,
            "response": f"Retrieval error: {exc}",
            "context": []
        }

    retrieved_context = [
        doc.page_content
        for doc in docs
    ]

    context = "\n".join(retrieved_context)

    # =========================
    # Medical Prompt
    # =========================

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

    # =========================
    # OpenRouter Call
    # =========================

    result = call_openrouter(prompt)

    if not result.get("success"):

        return {
            "severity": severity,
            "response": (
                f"OpenRouter error: "
                f"{result.get('error')}"
            ),
            "context": retrieved_context
        }

    data = result.get("data", {})

    try:

        content = data["choices"][0]["message"]["content"]

    except Exception:

        logger.exception(
            "Failed to extract OpenRouter response"
        )

        return {
            "severity": severity,
            "response": "Malformed OpenRouter response",
            "context": retrieved_context
        }

    return {
        "severity": severity,
        "response": content,
        "context": retrieved_context
    }
