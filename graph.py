from typing import TypedDict, Optional
import numpy as np
from langgraph.graph import StateGraph, START, END

from nodes.router import query_router_node, route_decision
from nodes.predictive import llm1_query_transformation, forecasting_ann_node, llm2_output_generation
from nodes.rag import rag_retriever_node, phi3_finetuned_node

class GraphState(TypedDict):
    query: str
    route: str
    numpy_array: Optional[np.ndarray]
    ann_output: Optional[dict]
    rag_context: Optional[str]
    final_response: str

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("query_router",        query_router_node)
    g.add_node("llm1_transformation", llm1_query_transformation)
    g.add_node("forecasting_ann",     forecasting_ann_node)
    g.add_node("llm2_output",         llm2_output_generation)
    g.add_node("rag_retriever",       rag_retriever_node)
    g.add_node("phi3_finetuned",      phi3_finetuned_node)

    g.add_edge(START, "query_router")
    g.add_conditional_edges("query_router", route_decision, {
        "predictive_analytics": "llm1_transformation",
        "general_query":        "rag_retriever",
    })
    g.add_edge("llm1_transformation", "forecasting_ann")       ########### 1st workflow branch
    g.add_edge("forecasting_ann",     "llm2_output")
    g.add_edge("llm2_output",         END)
    g.add_edge("rag_retriever",       "phi3_finetuned")        ########### 2nd workflow branch
    g.add_edge("phi3_finetuned",      END)

    return g.compile()

app = build_graph()