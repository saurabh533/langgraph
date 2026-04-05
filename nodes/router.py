from typing import Literal
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq



#-------------------------------------------
# SECURELY LOAD YOUR GROQ API KEY           
#--------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0)
# ─────────────────────────────────────────
# 2. ROUTER
# ─────────────────────────────────────────
class RouteDecision(BaseModel):
    destination: Literal["predictive_analytics", "general_query", "general"]

#llm = ChatOpenAI(model="gpt-4o", temperature=0)

router_chain = ChatPromptTemplate.from_messages([
    ("system", """Route the query:
- 'predictive_analytics': forecasting, predictions, trends,"Relative_Compactness","Surface_Area","Wall_Area","Roof_Area","Overall_Height","Orientation","Glazing_Area","Glazing_Area_Distribution"
- 'general_query': factual questions related to energy, explanations related to energy, domain knowledge or scheme related to energy, general energy information
-- "general": If user is asking hello, hi, or just wants to talk or chat very general things with you"""),
    ("human", "{query}")
]) | llm.with_structured_output(RouteDecision)

def query_router_node(state: dict) -> dict:                         ############## first node 
    decision = router_chain.invoke({"query": state["query"]})
    print(f"[ROUTER] Route decision: {decision.destination}")
    return {"route": decision.destination}

def route_decision(state: dict) -> str:
    return state["route"]
