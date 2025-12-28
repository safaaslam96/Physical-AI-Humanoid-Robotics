"""
Test script to verify Gemini API works with the stable model
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("[ERROR] GEMINI_API_KEY not found in .env")
    exit(1)

print(f"[OK] GEMINI_API_KEY loaded: {gemini_api_key[:10]}...")

# Test with stable model
genai.configure(api_key=gemini_api_key)

# First, list available models
print("\n[INFO] Listing available Gemini models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")
except Exception as e:
    print(f"[WARNING] Could not list models: {e}")

# Try gemini-1.5-flash
model = genai.GenerativeModel("gemini-1.5-flash")

print("\n[TEST] Testing Gemini API with gemini-1.5-flash model...")

try:
    response = model.generate_content(
        "What is ROS 2 in robotics? Explain briefly in 2-3 sentences.",
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=200,
            temperature=0.7
        )
    )

    print("\n[SUCCESS] Gemini API is working!")
    print(f"\n[RESPONSE]\n{response.text}")

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("\nThis might be a quota issue or wrong model name. Trying alternative models...")

    # Try gemini-pro as fallback
    try:
        print("\n[TEST] Trying gemini-pro model...")
        model_fallback = genai.GenerativeModel("gemini-pro")
        response = model_fallback.generate_content(
            "What is ROS 2 in robotics? Explain briefly in 2-3 sentences.",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.7
            )
        )
        print("\n[SUCCESS] gemini-pro model is working!")
        print(f"\n[RESPONSE]\n{response.text}")
        print("\n[NOTE] Use 'gemini-pro' instead of 'gemini-1.5-flash' in config.py")
    except Exception as e2:
        print(f"\n[ERROR] gemini-pro also failed: {e2}")
