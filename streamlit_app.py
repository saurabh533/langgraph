# streamlit_app.py
import streamlit as st

# ── Load graph once (cached so it doesn't reload on every interaction) ────────
@st.cache_resource
def load_graph():
    from graph import app
    return app

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 LangGraph Chatbot")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": "..."}]

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything…"):

    # Show & store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the graph
    with st.chat_message("assistant"):
        # Show live node progress while graph is running
        status_box = st.status("Thinking…", expanded=False)
        response_placeholder = st.empty()
        final_result = None
        workflow_name = None
        graph = load_graph()

        for step in graph.stream({"query": prompt}, stream_mode="updates"):
            for node_name, output in step.items():

                # ✅ Capture FIRST node only
                if workflow_name is None:
                   workflow_name = output

                # ✅ Capture final response (don't display yet)
                if "final_response" in output:
                   final_result = output["final_response"]
       
        # ✅ Show workflow FIRST
        if workflow_name:
           st.markdown(f"**Workflow selected:** `{workflow_name['route']}`")

        # ✅ Then show response
        if final_result:
           st.markdown(final_result)
        else:
           st.markdown("_(No final_response found in graph output)_")

        status_box.update(label="Done", state="complete", expanded=False)

    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": final_result})
