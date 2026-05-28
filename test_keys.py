import os
import requests
from dotenv import load_dotenv

load_dotenv()

TEST_MODELS = [
    "openrouter/free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "google/gemma-3-27b-it:free",
]

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

working_keys = []
failed_keys = []

for i in range(11, 13):

    key = os.getenv(f"OPENROUTER_API_KEY_{i}")

    if not key:
        print(f"[KEY {i}] Missing")
        continue

    print(f"\n==============================")
    print(f"Testing KEY {i}")
    print(f"==============================")

    key_worked = False

    for model in TEST_MODELS:

        print(f"Trying model: {model}")

        try:

            response = requests.post(
                BASE_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello"
                        }
                    ]
                },
                timeout=30
            )

            print("Status:", response.status_code)

            try:
                data = response.json()
            except:
                print("Invalid JSON")
                continue

            if response.status_code == 200 and "choices" in data:

                print("SUCCESS")
                print("Working model:", model)

                working_keys.append({
                    "key_index": i,
                    "model": model
                })

                key_worked = True
                break

            else:
                print("FAILED")
                print(data)

        except Exception as e:
            print("EXCEPTION:", str(e))

    if not key_worked:
        failed_keys.append(i)

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print("\nWORKING KEYS:")
for item in working_keys:
    print(item)

print("\nFAILED KEYS:")
print(failed_keys)