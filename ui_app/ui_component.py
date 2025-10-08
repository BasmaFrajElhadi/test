import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from groq import Groq
from google import genai
import streamlit as st
from rag.corrective_rag import CorrectiveRAG
from core.sqlite_chat_storage import SQLiteChatStorage
from google.api_core.exceptions import PermissionDenied, InvalidArgument, Unauthenticated


class UIComponent:
    def __init__(self, store):
        self.store = store
        self._init_session_state()

    def _init_session_state(self):
        """
        Initializes Streamlit session state: deletes empty sessions, 
        creates a new chat session, and sets up messages if not already set.
        """
        if "current_session" not in st.session_state:
            # delete empty sessions when reload 
            self.store.delete_empty_sessions()
            # create a fresh chat session when app starts
            new_id = self.store.time_random_id()
            self.store.create_session(new_id, "new chat")
            st.session_state.current_session = new_id
            st.session_state.messages = []


    def validate_gemini_key(self, key: str) -> bool:
        """
        Checks if a Google AI (Gemini) API key is valid.

        Tries to access the "gemini-2.5-flash" model using the provided key.
        Returns True if successful, otherwise False.

        Parameters:
            key (str): Google AI API key.

        Returns:
            bool: True if key is valid, False otherwise.
        """
        try:
            client = genai.Client(api_key=key)
            response  = client.models.get(model = "gemini-2.5-flash",)
            return True
        except (PermissionDenied, InvalidArgument, Unauthenticated) as e:
            print(f"Validation failed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during validation: {e}")
            return False
 
    def validate_groq_key(self, key: str) -> bool:
        """
        Checks if a Groq API key is valid.

        Tries to create a simple chat completion using the provided key.
        Returns True if successful, otherwise False.

        Parameters:
            key (str): Groq API key.

        Returns:
            bool: True if key is valid, False otherwise.
        """
        try:

            client = Groq(api_key=key)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "ping"}],
                model="groq/compound"
            )
            return True
        except Exception as e:
            print(f"Groq key validation failed: {e}")
            return False

    def start_new_session(self):
        """
        Starts a new chat session: deletes empty current session, 
        creates a new session, and resets messages.
        """
            
        # delete empty chat before starting new one
        if len(st.session_state.messages) == 0:
            self.store.delete_session(st.session_state.current_session)
        new_id = self.store.time_random_id()
        self.store.create_session(new_id, "new chat")
        st.session_state.current_session = new_id
        st.session_state.messages = []

    def display_previous_sessions(self, session, store):
        """
        Displays a sidebar button for a previous session. 
        Updates current session and messages if clicked.
        
        Parameters:
        - session (dict): Session info with "id" and optional "name".
        - store (object): Provides `get_messages(session_id)` to load session messages.
        """
        active_button = "secondary"
        if st.session_state.current_session == session["id"]:
            active_button = "primary"
        label = session["name"] or session["id"]
        if st.sidebar.button(label, key=session["id"],width="stretch",type=active_button):
            st.session_state.current_session = session["id"]
            st.session_state.messages = store.get_messages(session["id"])
            st.rerun() 

    def get_pdf_path(self, university_name: str):
        """
        Returns the file path for a university's PDF document.

        Parameters:
        - university_name (str): Name of the university.

        Returns:
        - str: Full path to the PDF file in the 'data/docs' folder.
        """
        folder_path = os.path.join(project_root, "data", "docs")
        safe_name = university_name.replace(' ', '_')
        filename = f"{safe_name}.pdf"
        return os.path.join(folder_path, filename)
    

