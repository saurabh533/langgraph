# langgraph
All in one Energy Solution Chatbot.
AI Query Routing System An intelligent AI application built with Streamlit and LangGraph that dynamically routes user queries to the most appropriate AI workflow, including a general-purpose LLM, a pre-trained ANN model, or a Retrieval-Augmented Generation (RAG) pipeline. 
Features 
• Interactive web interface with Streamlit 
• LangGraph-based routing architecture 
• General LLM for conversational queries 
• Pre-trained ANN model for predictions 
• RAG pipeline with vector database retrieval 
• Modular and scalable workflow 

Project Structure 
├── app.py # Streamlit application 
├── graph.py # LangGraph routing workflow 
├── train_ann.py # ANN model training 
├── requirements.txt # Project dependencies 
└── README.md 

Installation 
git clone https://github.com/<saurabh533>/<langgraph>.git 
cd <repository> 
pip install -r requirements.txt 
streamlit run app.py 

Workflow 
    User Query 
         │ 
         ▼ 
LangGraph Router 
    ├── General LLM 
    ├── ANN Model 
    └── RAG + Vector Database 
         │ 
         ▼ 
  Final Response 
  
  Tech Stack 
  • Python 
  • Streamlit
  • LangGraph 
  • ANN 
  • RAG 
  • Vector Database License 
  This project is intended for educational and research purposes.
