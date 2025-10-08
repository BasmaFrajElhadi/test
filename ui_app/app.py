import sys
import os
import streamlit as st
# --- Project path setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from rag.corrective_rag import CorrectiveRAG
from core.vector_db import VectorDB
from core.sqlite_chat_storage import SQLiteChatStorage
from ui_app.ui_component import UIComponent

# --- Page setup ---
store = SQLiteChatStorage()
ui_component = UIComponent(store)
config = {"configurable": {"thread_id": "1"}}
st.set_page_config(
    page_title="Egyptian Universities Assistant",
    page_icon="🧑‍🎓",
    layout="wide"
)
validate_gemini_key = False
validate_groq_key = False

# --- Initialize once ---
@st.cache_resource
def get_vector_db():
    db = VectorDB()
    db.create_collection()
    db.add_to_collection()
    return db
vector_db = get_vector_db()

# --- Sidebar ---

# --- Google AI Key ---
# --- Google AI Key ---
st.sidebar.markdown("### 🔑 Google AI Key")
google_key_input = st.sidebar.text_input(
    label="",
    label_visibility="hidden",
    type="password",
    key="google_ai_key_input"
)

validate_gemini_key = ui_component.validate_gemini_key(google_key_input)

if google_key_input:
    if validate_gemini_key:
        st.sidebar.success("✅ Key is valid!")
    else:
        st.sidebar.error("❌ Invalid key. Please check and try again.")


# --- Groq Key ---
st.sidebar.markdown("### 🔑 Groq Key")
groq_key_input = st.sidebar.text_input(
    label="",
    label_visibility="hidden",
    type="password",
    key="groq_key_input"  # unique key
)

validate_groq_key = ui_component.validate_groq_key(groq_key_input)

if groq_key_input:
    if validate_groq_key:
        st.sidebar.success("✅ Key is valid!")
    else:
        st.sidebar.error("❌ Invalid key. Please check and try again.")

if validate_gemini_key and validate_groq_key:
    corrective_rag = CorrectiveRAG(google_key_input, groq_key_input)
    compiled_graph = corrective_rag.build_graph()


if st.sidebar.button("➕ New Chat",type="primary",width="stretch"):
    ui_component.start_new_session()
    

# Display previous sessions
sessions = store.list_sessions()
st.sidebar.markdown("### 💬 Previous Chats")

if sessions:
    for session in sessions:
        ui_component.display_previous_sessions(session, store)
else:
    st.sidebar.markdown("#### No chat history yet")

# --- Main area ---
st.markdown("<h1 style='text-align:center;'>🔍 University Information Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>Learn about public universities in Egypt</h2>", unsafe_allow_html=True)

# Display existing chat
for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = "🧑‍🎓"
    else: 
        avatar = "🎓"
    with st.chat_message(message["role"] , avatar = avatar):
        st.markdown(message["content"])

# Handle new user message
if validate_gemini_key and validate_groq_key:
    if prompt := st.chat_input("Enter your question here"):
        session_id = st.session_state.current_session

        # Display user message
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        # If this is the first message in the session,
        # use its content to automatically set the session name
        if len(st.session_state.messages) == 0:
            store.update_session_name(session_id, prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        store.add_user_message(session_id, prompt)

        # RAG response
        with st.spinner("🔎 Searching the knowledge base..."):
            # result = compiled_graph.invoke({"query": prompt}, config=config)
            # response = result["agent_response"]
            # agent_metadata = result["agent_metadata"]

            try:
                result = compiled_graph.invoke({"query": prompt}, config=config)
                response = result.get("agent_response", "I can’t understand the question.")
                agent_metadata = result.get("agent_metadata", {})
            except Exception as e:
                # Log the error (optional)
                print("Invoke error:", e)
                response = "I can’t understand the question."
                agent_metadata = {}

        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(response)

        if agent_metadata:

            full_file_path = ui_component.get_pdf_path(agent_metadata['university_name'][0])
            st.pdf(full_file_path, height=700)
            
            try:
                st.pdf(full_file_path, height=700)
            except Exception as e:
                pass

        st.session_state.messages.append({"role": "assistant", "content": response})
        store.add_ai_message(session_id, response,agent_metadata)
else:
    st.chat_input("Enter your question here", disabled=True)