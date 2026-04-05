import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import shap
import tensorflow as tf
import pandas as pd
from pydantic import BaseModel







# ── Load model + SHAP background ONCE at module level ─────────────────────────
print("[ANN] Loading Keras model...")
ann_model = tf.keras.models.load_model("best_heating_load_model.keras")
print(f"[ANN] Input shape: {ann_model.input_shape}")   # confirm (None, 8)
print(f"[ANN] Output shape: {ann_model.output_shape}") # confirm (None, 1)

print("[SHAP] Loading background data...")
background = pd.read_pickle("shap_background.pkl")
background_np = background.values.astype(np.float32)  # DeepExplainer needs float32

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



#─────────────────────────────────────────
#Class for structured output
#------------------------------
class EnergyFeatures(BaseModel):
    Relative_Compactness: float
    Surface_Area: float
    Wall_Area: float
    Roof_Area: float
    Overall_Height: float
    Orientation: float
    Glazing_Area: float
    Glazing_Area_Distribution: float



#-------------------------------------------
# SECURELY LOAD YOUR GROQ API KEY           
#--------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0)




# ─────────────────────────────────────────
# 3. PREDICTIVE ANALYTICS BRANCH
# ─────────────────────────────────────────

llm1_chain = ChatPromptTemplate.from_messages([
    ("system", """You are a feature extraction assistant for an energy efficiency ANN model.
Extract exactly these 8 building features from the user query in order:
    I am showing you example user input 
- Relative_Compactness     (e.g. 0.62 – 0.98)
- Surface_Area             (e.g. 514.5 – 808.5 m²)
- Wall_Area                (e.g. 245.0 – 416.5 m²)
- Roof_Area                (e.g. 110.25 – 220.5 m²)
- Overall_Height           (e.g. 3.5 or 7.0 m)
- Orientation              (e.g. 2, 3, 4, or 5)
- Glazing_Area             (e.g. 0.0 – 0.4)
- Glazing_Area_Distribution(e.g. 0, 1, 2, 3, 4, or 5)

Output only the list of 8 values, in this format:[0.764, 671.7, 318.5, 176.6, 5.25, 3.5, 0.234, 2.812]"""),
    ("human", "{query}")
]) | llm.with_structured_output(EnergyFeatures)



def llm1_query_transformation(state: dict) -> dict:
    print("[LLM1] Extracting building features from query...")
    response: EnergyFeatures = llm1_chain.invoke({"query": state["query"]})         ##########node 2nd

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
    arr = state["numpy_array"]               # shape (8,)
    input_2d = arr.reshape(1, -1)            # shape (1, 8) ← Keras needs this

    # Prediction
    raw_output = ann_model.predict(input_2d, verbose=0)  # shape (1, 1)
    prediction = float(raw_output[0][0])                                    ###########node 3
    print(f"[ANN] Predicted heating load: {prediction:.4f} kWh/m²")

    # SHAP — DeepExplainer also needs 2D input
    print("[SHAP] Computing attributions...")
    shap_values = explainer.shap_values(input_2d)
    # shap_values is a list with one array of shape (1, 8)
    shap_array = np.array(shap_values).flatten()  # flatten to (8,)
    # Named feature attribution dict
    shap_dict = {
        name: round(float(val), 6)
        for name, val in zip(FEATURE_NAMES, shap_array)
    }

    # Top 3 drivers with names (not indices)
    top_features = sorted(
        shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )[:3]

    print(f"[SHAP] Top drivers: {top_features}")

    return {
        "ann_output": {
            "prediction": round(prediction, 4),
            "shap_dict": shap_dict,
            "top_features": top_features,    # [(feature_name, shap_value), ...]
        }
    }


# ── LLM 2 output generation ───────────────────────────────────────────────────
llm2_chain = ChatPromptTemplate.from_messages([
    ("system", """You are an energy efficiency expert. Explain the building's 
predicted heating load and what physical features drove the result, 
in plain language for an architect or building manager."""),
    ("human", """Original query: {query}

Predicted Heating Load: {prediction} kWh/m²

Top 3 features influencing this prediction:
{top_features}

Explain what the prediction means and why these features had the most impact.""")
]) | llm


def llm2_output_generation(state: dict) -> dict:
    print("[LLM2] Generating natural language response...")
    ann = state["ann_output"]

    # Format top features with direction (increases/decreases load)
    top_formatted = "\n".join([
        f"  - {name}: SHAP {val:+.4f} "
        f"({'increases' if val > 0 else 'decreases'} heating load)"
        for name, val in ann["top_features"]
    ])
                                                                                ############node 4 in first workflow
    response = llm2_chain.invoke({
        "query": state["query"],
        "prediction": ann["prediction"],
        "top_features": top_formatted,
    })
    return {"final_response": response.content}