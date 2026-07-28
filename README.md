# All-in-One Energy Solution Chatbot
 
A chatbot for energy-related questions that figures out *how* to answer you instead of just throwing everything at one model.
 
Some questions are conversational, some need a quick numeric prediction, and some only make sense if you pull facts from actual documents. Cramming all of that into a single LLM call tends to give mediocre answers across the board. So this project uses a router (built with LangGraph) that looks at your question first and sends it to whichever workflow fits best — a general LLM, a trained ANN, or a RAG pipeline backed by a vector store.
 
The front end is a simple Streamlit app, so you can just type and go.
 
## Why the routing matters
 
The three paths handle very different kinds of questions:
 
- **General LLM** — for open-ended or conversational stuff, like *"What are some cheap ways to cut my electricity bill?"*
- **ANN model** — when you actually want a number back, like predicting next month's consumption.
- **RAG + vector database** — when the answer should come from your own documents (reports, policies, datasets) rather than the model's general knowledge.
Routing keeps each part focused. The LLM doesn't try to do math it's bad at, and the ANN doesn't try to hold a conversation.
 
## What's in here
 
```
app.py            # the Streamlit app you actually run
graph.py          # the LangGraph router + the three workflow nodes
train_ann.py      # script to train the ANN model
requirements.txt  # dependencies
```
 
## Running it
 
You'll need Python 3.10+ and an API key for whatever LLM provider you're using.
 
```bash
git clone https://github.com/<saurabh533>/<langgraph>.git
cd <langgraph>
 
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
 
pip install -r requirements.txt
```
 
If you want predictions to work, train the ANN once:
 
```bash
python train_ann.py
```
 
Then start the app:
 
```bash
streamlit run app.py
```
 
It should open at `http://localhost:8501`.
 
## API keys
 
Don't hard-code keys. Drop them in a `.env` file (and make sure it's in `.gitignore`):
 
```env
OPENAI_API_KEY=your_key_here
VECTOR_DB_PATH=./vectorstore
```
 
Rename the variables to match whatever provider and vector store you settled on.
 
## How a query flows through it
 
```
User query
   │
   ▼
LangGraph router  ──►  General LLM
                  ──►  ANN model
                  ──►  RAG + vector database
   │
   ▼
Answer
```
 
The router is the only piece that decides where things go, so adding a fourth workflow later is mostly a matter of adding a node and a branch.
 
## Built with
 
Python, Streamlit, LangGraph, a small ANN for predictions, and a RAG setup on top of a vector database.
 
## Notes
 
This was put together for learning and experimentation, so treat it as a starting point rather than something production-ready. A few things I'd still like to add: conversation memory between turns, a bit of evaluation to check how often the router picks the right path, and a Docker setup so it's easier to run.
 
## License
 
Free to use for educational and research purposes.
