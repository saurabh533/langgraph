# nodes/rag.py
import os
import torch
import numpy as np
from astrapy import DataAPIClient
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ── 1. AstraDB client via astrapy ─────────────────────────────────────────────
client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
db = client.get_database_by_api_endpoint(os.environ["ASTRA_DB_API_ENDPOINT"])
collection = db.get_collection(os.environ["ASTRA_COLLECTION"])

# ── 2. Embedding model ────────────────────────────────────────────────────────
print("[EMB] Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── 3. Phi-3 fine-tuned model (commented out — disk space) ───────────────────
# MODEL_BASE = "microsoft/Phi-3-mini-4k-instruct"
# PEFT_PATH  = "D:\\Chatbot-energy\\rags\\rags"
# print("[PHI3] Loading base model...")
# tokenizer = AutoTokenizer.from_pretrained(PEFT_PATH, trust_remote_code=True)
# base_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_BASE,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True,
# )
# print("[PHI3] Applying PEFT weights...")
# model = PeftModel.from_pretrained(base_model, PEFT_PATH)
# model.eval()
# phi3_pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=False,
#     return_full_text=False,
# )
# print("[PHI3] Model ready.")

# ── 4. Groq LLM (replacing Phi-3) ────────────────────────────────────────────
print("[GROQ] Initialising Groq LLM...")
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",       # fast + free tier available
    # model="llama-3.3-70b-versatile",  # swap for better quality if needed
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
)

groq_chain = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in the energy domain with deep knowledge of:
- Smart grids and demand response
- Energy efficiency and building energy systems
- Renewable energy integration
- Power systems and load forecasting
- Energy policy and sustainability

Answer questions using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Keep answers concise, accurate, and technical where appropriate."""),
    ("human", """Context:
{context}

Question: {query}

Answer:""")
]) | groq_llm

print("[GROQ] Ready.")


# ── 5. RAG retriever node ─────────────────────────────────────────────────────
def rag_retriever_node(state: dict) -> dict:
    print("[RAG] Embedding query...")
    query_vector = embedder.encode(state["query"]).tolist()

    print("[RAG] Querying AstraDB collection...")
    results = collection.find(
        sort={"$vector": query_vector},
        limit=4,                                                       ####node 3 
        projection={"text": 1, "source": 1, "$vector": 0},
    )

    docs = list(results)
    if not docs:
        print("[RAG] No results found, proceeding with empty context.")
        context = "No relevant context found."
    else:
        context = "\n\n".join([
            f"[Source {i+1} | {doc.get('source', 'unknown')}]:\n{doc.get('text', '')}"
            for i, doc in enumerate(docs)
        ])
        print(f"[RAG] Retrieved {len(docs)} chunks.")

    return {"rag_context": context}


# ── 6. Groq inference node (replaces phi3_finetuned_node) ────────────────────
def phi3_finetuned_node(state: dict) -> dict:
    # kept the same function name so graph.py needs zero changes
    print("[GROQ] Generating response...")

    response = groq_chain.invoke({
        "context": state["rag_context"],
        "query": state["query"],                                   #############node 4 in 2nd workflow
    })

    result = response.content.strip()
    print(f"[GROQ] Done. ({len(result)} chars)")
    return {"final_response": result}
