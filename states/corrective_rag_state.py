from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from typing_extensions import TypedDict, NotRequired, Optional

class CorrectiveRAGState(TypedDict):
    """ CorrectiveRAGState state for the RAG system. """
    query: str
    agent_response: str
    vector_store: Chroma
    relevant_documents: list[str]
    filter_model: ChatGoogleGenerativeAI
    basic_model: ChatGoogleGenerativeAI
    agent_metadata: NotRequired[Optional[dict]]


