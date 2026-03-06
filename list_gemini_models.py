import os
from dotenv import load_dotenv
import google.generativeai as genai

print("Before load_dotenv, GEMINI_API_KEY =", repr(os.environ.get("GEMINI_API_KEY")))

load_dotenv()  # loads .env if present

print("After load_dotenv, GEMINI_API_KEY =", repr(os.environ.get("GEMINI_API_KEY")))

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("No GEMINI_API_KEY set")

genai.configure(api_key=key)

for m in genai.list_models():
    if "generateContent" in getattr(m, "supported_generation_methods", []):
        print(m.name)