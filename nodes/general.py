# nodes/general.py  — small talk / general chat (balanced model: gemini-2.5-flash)
import os
from langchain_google_vertexai import ChatVertexAI

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION   = os.environ.get("GCP_LOCATION", "us-central1")

llm = ChatVertexAI(
    model="gemini-2.5-pro",
    temperature=0.3,          # a little warmth for greetings/small talk
    project=PROJECT_ID,
    location=LOCATION,
)

def general_energydomain_llm(state: dict) -> dict:
    query = state["query"]
    response = llm.invoke([
        {
            "role": "system",
            "content": (
                "You are a friendly conversational assistant. "
                "You specialize in energy domain and predictive analytics topics. "
                "For general greetings and small talk, respond warmly and briefly. "
                "Always guide the user towards asking about energy or analytics if relevant."
            ),
        },
        {"role": "user", "content": query},
    ])
    return {"final_response": response.content}