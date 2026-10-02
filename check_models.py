import os
from google import genai
1
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

for model in client.models.list():
    if 'generateContent' in model.supported_actions:
        print(model.name)