import os
import sys
from google import genai
from dotenv import load_dotenv
from langsmith.run_helpers import traceable

# --- Project path setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from states.conversation_state import ConversationState
from core.vector_db import VectorDB
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


class FoundationRAG:
    """
    FoundationRAG (Retrieval-Augmented Generation) integrates a vector database for information retrieval
    and Google's generative AI models (Gemini) for contextual answer generation.

    It follows the RAG architecture:
        1. Retrieval: Fetch relevant chunks from a vector database.
        2. Augmentation: Build a prompt that includes retrieved context.
        3. Generation: Use an LLM to produce a grounded, informative answer.
    """

    def __init__(self,google_api_key):
        """
        Initialize the RAG system:
        - Connects to the vector database (ChromaDB or equivalent).
        - Ensures the collection is ready and populated.
        - Initializes Google Generative AI client.
        """
        self.google_api_key = google_api_key
        self.vector_db = VectorDB()
        self.collection = self.vector_db.create_collection()
        self.client = genai.Client(api_key=self.google_api_key)

        # Initialize collection if empty
        if self.collection is not None and self.collection.count() == 0:
            print("Adding initial documents to ChromaDB")
            self.vector_db.add_to_collection()
        else:
            print(f"Collection already has {self.collection.count()} documents. Skipping add.")

    @traceable
    def retrieval(self, query, k=5):
        """
        Retrieve top-k most relevant documents from the vector database for a given query.

        Args:
            query (str): User's input query.
            k (int, optional): Number of top documents to return. Defaults to 5.

        Returns:
            list[Document]: List of LangChain `Document` objects containing 
                            the retrieved text and metadata.
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents:
            return []
        
        # Combine retrieved texts and their corresponding metadata
        retrieved_docs = [
            Document(page_content=documents[0][i], metadata=metadatas[0][i])
            for i in range(len(documents[0]))
        ]
        return retrieved_docs

    @traceable
    def augmented(self, query, retrieved_documents):
        """
        Construct an augmented prompt by combining the user's query with the retrieved context.

        Args:
            query (str): User's question.
            retrieved_documents (list[Document]): Documents fetched from the vector store.

        Returns:
            str: Fully formatted prompt ready to be passed to the LLM.
        """
        # Combine the retrieved text into a single context block
        context = "\n\n".join([doc.page_content for doc in retrieved_documents])

        prompt = f"""
        You are an expert on Egyptian public universities. 
        Use the following document excerpts to answer the user query. 
        Do not make assumptions; only provide information present in the documents.

        Document Context:
        {context}

        Instructions:
        - Answer clearly and concisely.
        - Include any relevant details like faculty names, contact info, admission requirements, or location if available in the context.
        - Keep your answer in a readable paragraph format.
        """
        return prompt

    @traceable
    def generation(self, state: ConversationState, retrieved_documents, model_name="gemini-2.5-flash") -> ConversationState:
        """
        Generate an answer to the user query using the augmented context and Gemini LLM.

        Args:
            state (ConversationState): Object holding the conversation's message.
            retrieved_documents (list[Document]): Relevant documents for context.
            model_name (str, optional): Google Generative AI model name. Defaults to "gemini-2.5-flash".

        Returns:
            dict: Updated conversation state containing the model's response message.
        """
        # Extract the most recent user query
        query = state['messages'][-1].content

        prompt = self.augmented(query, retrieved_documents)

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ]

        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7, api_key=self.google_api_key)
        response = llm.invoke(messages)

        return {"messages": [response]}

