"""
Test gemini-2.5-flash model
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

print("[TEST] Testing gemini-2.5-flash model...")

try:
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        "What is ROS 2 in robotics? Explain briefly in 2-3 sentences.",
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=200,
            temperature=0.7
        )
    )

    print("\n[SUCCESS] gemini-2.5-flash is working!")
    print(f"\n[RESPONSE]\n{response.text}")

except Exception as e:
    print(f"\n[ERROR] {e}")
