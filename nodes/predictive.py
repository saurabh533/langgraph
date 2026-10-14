# nodes/predictive.py
import os
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
import shap
import tensorflow as tf
import pandas as pd
from pydantic import BaseModel

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION   = os.environ.get("GCP_LOCATION", "us-central1")

# ── Load model + SHAP background ONCE at module level ─────────────────────────
print("[ANN] Loading Keras model...")
ann_model = tf.keras.models.load_model("best_heating_load_model.keras")
print(f"[ANN] Input shape: {ann_model.input_shape}")    # confirm (None, 8)
print(f"[ANN] Output shape: {ann_model.output_shape}")  # confirm (None, 1)

print("[SHAP] Loading background data...")
background = pd.read_pickle("shap_background.pkl")
background_np = background.values.astype(np.float32)    # DeepExplainer needs float32

print("[SHAP] Building DeepExplainer...")
explainer = shap.DeepExplainer(ann_model, background_np)
print("[SHAP] Explainer ready.")

FEATURE_NAMES = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution",
]

# ── Structured output schema ──────────────────────────────────────────────────
class EnergyFeatures(BaseModel):
    Relative_Compactness: float
    Surface_Area: float
    Wall_Area: float
    Roof_Area: float
    Overall_Height: float
    Orientation: float
    Glazing_Area: float
    Glazing_Area_Distribution: float

# ── LLM1: feature extraction  (average tier: gemini-2.5-flash, deterministic) ──
extract_llm = ChatVertexAI(
    model="gemini-2.5-flash",
    temperature=0,
    project=PROJECT_ID,
    location=LOCATION,
)

# ── LLM2: paraphrase final output to user (cheapest: gemini-2.5-flash-lite) ───
paraphrase_llm = ChatVertexAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    project=PROJECT_ID,
    location=LOCATION,
)

# ─────────────────────────────────────────
# 3. PREDICTIVE ANALYTICS BRANCH
# ─────────────────────────────────────────
# NOTE: for Gemini, drop method="json_mode" — Gemini enforces the schema via
# controlled generation (more reliable than JSON mode).
llm1_chain = ChatPromptTemplate.from_messages([
    ("system", """You are a feature extraction assistant for an energy efficiency ANN model.
Extract exactly these 8 building features from the user query.

Return a JSON object with EXACTLY these keys (all numbers, no lists, no arrays):
- Relative_Compactness (e.g. 0.62 - 0.98)
- Surface_Area (m2, e.g. 514.5 - 808.5)
- Wall_Area (m2, e.g. 245.0 - 416.5)
- Roof_Area (m2, e.g. 110.25 - 220.5)
- Overall_Height (m, e.g. 3.5 or 7.0)
- Orientation (2, 3, 4, or 5)
- Glazing_Area (e.g. 0.0 - 0.4)
- Glazing_Area_Distribution (0, 1, 2, 3, 4, or 5)

Example output:
{{"Relative_Compactness": 0.764, "Surface_Area": 671.7, "Wall_Area": 318.5, "Roof_Area": 176.6, "Overall_Height": 5.25, "Orientation": 3, "Glazing_Area": 0.234, "Glazing_Area_Distribution": 2}}"""),
    ("human", "{query}")
]) | extract_llm.with_structured_output(EnergyFeatures)

def llm1_query_transformation(state: dict) -> dict:
    print("[LLM1] Extracting building features from query...")
    response: EnergyFeatures = llm1_chain.invoke({"query": state["query"]})   # node 2

    # Preserve exact feature order your ANN was trained on
    numpy_array = np.array([
        response.Relative_Compactness,
        response.Surface_Area,
        response.Wall_Area,
        response.Roof_Area,
        response.Overall_Height,
        response.Orientation,
        response.Glazing_Area,
        response.Glazing_Area_Distribution,
    ], dtype=np.float32)

    print(f"[LLM1] Extracted features: {numpy_array}")
    return {"numpy_array": numpy_array}

# ── Forecasting ANN + SHAP node ───────────────────────────────────────────────
def forecasting_ann_node(state: dict) -> dict:
    print("[ANN] Running inference...")
    arr = state["numpy_array"]        # shape (8,)
    input_2d = arr.reshape(1, -1)     # shape (1, 8)  Keras needs this

    raw_output = ann_model.predict(input_2d, verbose=0)   # shape (1, 1)
    prediction = float(raw_output[0][0])                  # node 3
    print(f"[ANN] Predicted heating load: {prediction:.4f} kWh/m2")

    print("[SHAP] Computing attributions...")
    shap_values = explainer.shap_values(input_2d)
    shap_array = np.array(shap_values).flatten()          # flatten to (8,)
    shap_dict = {
        name: round(float(val), 6)
        for name, val in zip(FEATURE_NAMES, shap_array)
    }

    top_features = sorted(
        shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )[:3]
    print(f"[SHAP] Top drivers: {top_features}")

    return {
        "ann_output": {
            "prediction": round(prediction, 4),
            "shap_dict": shap_dict,
            "top_features": top_features,
        }
    }

# ── LLM 2 output generation ───────────────────────────────────────────────────
llm2_chain = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly energy efficiency expert explaining a prediction to an \
everyday person - a homeowner or someone curious about buildings, with no technical or \
construction background.

Your explanation must:
- Start with one simple sentence stating the predicted heating load and whether that is \
low, moderate, or high for a normal building give the output prediction value.
- Explain what heating load means in everyday terms: how much energy the building needs to \
stay warm - so a higher number means it's harder (and more expensive) to heat, and a lower \
number means it's cozier and cheaper to keep warm.
- Go through each of the top 3 features in order. For each, say in plain words what it is \
(using a simple comparison or picture where it helps) and why it makes heating easier or \
harder - for example, "a bigger outer surface is like a cup with no lid: heat escapes faster."
- End with 1-2 simple, practical takeaways an ordinary person could understand \
(e.g. more windows, building shape, which way it faces) - as ideas to consider, not promises.

Rules:
- Use warm, everyday language, like explaining to a friend. Avoid jargon completely; if a \
real-world term is unavoidable, explain it right away in the same sentence.
- Use short sentences and simple analogies. Assume zero knowledge of construction or energy.
- Do NOT invent feature values, rankings, or numbers you were not given. Explain only the \
features and prediction provided.
- Do NOT mention the machine-learning model, algorithms, or any math.
- Keep it short and easy to read - a brief intro, the 3 points, and a friendly closing."""),
    ("human", """Original query: {query}

Predicted Heating Load: {prediction} kWh/m2

Top 3 features influencing this prediction (most to least impactful):
{top_features}

Explain what the prediction means and why these features had the most impact.""")
]) | paraphrase_llm

def llm2_output_generation(state: dict) -> dict:
    print("[LLM2] Generating natural language response...")
    ann = state["ann_output"]

    top_formatted = "\n".join([
        f" - {name}: SHAP {val:+.4f} "
        f"({'increases' if val > 0 else 'decreases'} heating load)"
        for name, val in ann["top_features"]
    ])
    response = llm2_chain.invoke({
        "query": state["query"],
        "prediction": ann["prediction"],
        "top_features": top_formatted,
    })
    return {"final_response": response.content}