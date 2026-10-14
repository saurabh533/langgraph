# nodes/rag.py  — Firestore vector search + Vertex AI (gemini-embedding-001 @ 768)
# RAG answer model: gemini-2.5-flash @ temperature 0 (deterministic retriever).
# No AstraDB, no sentence-transformers, no torch. Auth via the service account.
import os

import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID           = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION             = os.environ.get("GCP_LOCATION", "us-central1")
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "energy_docs")
FIRESTORE_DATABASE   = os.environ.get("FIRESTORE_DATABASE", "default")  # your DB's ID
EMBED_MODEL          = "gemini-embedding-001"
EMBED_DIM            = 768          # MUST match the Firestore vector index + ingestion
VECTOR_FIELD         = "embedding"
TOP_K                = 4

# ── Vertex AI init + embedding model ──────────────────────────────────────────
vertexai.init(project=PROJECT_ID, location=LOCATION)
_embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)

def embed_query(text: str) -> list[float]:
    # task_type RETRIEVAL_QUERY tells the model this is a search query (better recall).
    inp = TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY")
    res = _embed_model.get_embeddings([inp], output_dimensionality=EMBED_DIM)
    return res[0].values

# ── Firestore client (no keys — uses the service account) ─────────────────────
db = firestore.Client(project=PROJECT_ID, database=FIRESTORE_DATABASE)
collection = db.collection(FIRESTORE_COLLECTION)

# ── RAG answer model: deterministic ───────────────────────────────────────────
print("[LLM] Initialising Vertex AI Gemini (RAG)...")
rag_llm = ChatVertexAI(
    model="gemini-2.5-flash",
    temperature=0,
    project=PROJECT_ID,
    location=LOCATION,
)

rag_chain = ChatPromptTemplate.from_messages([
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
]) | rag_llm

# ── RAG retriever node (Firestore vector search) ──────────────────────────────
def rag_retriever_node(state: dict) -> dict:
    print("[RAG] Embedding query (gemini-embedding-001)...")
    query_vector = embed_query(state["query"])

    print("[RAG] Querying Firestore vector search...")
    snapshots = collection.find_nearest(
        vector_field=VECTOR_FIELD,
        query_vector=Vector(query_vector),
        distance_measure=DistanceMeasure.COSINE,
        limit=TOP_K,
    ).get()

    docs = [s.to_dict() for s in snapshots]
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

# ── Generation node (same function name so graph.py is untouched) ─────────────
def phi3_finetuned_node(state: dict) -> dict:
    print("[LLM] Generating response (Gemini)...")
    response = rag_chain.invoke({
        "context": state["rag_context"],
        "query": state["query"],
    })
    result = response.content.strip()
    print(f"[LLM] Done. ({len(result)} chars)")
    return {"final_response": result}