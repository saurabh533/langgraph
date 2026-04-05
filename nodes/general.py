# nodes/general_chat.py
from langchain_groq import ChatGroq
import os

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

def general_energydomain_llm(state: dict)-> dict:
    query = state["query"]
    
    response = llm.invoke([
        {
            "role": "system",
            "content": (
                "You are a friendly conversational assistant. "
                "You specialize in energy domain and predictive analytics topics. "
                "For general greetings and small talk, respond warmly and briefly. "
                "Always guide the user towards asking about energy or analytics if relevant."
            )
        },
        {
            "role": "user",
            "content": query
        }
    ])
    
    return {"final_response": response.content}