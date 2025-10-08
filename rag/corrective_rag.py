import os 
import sys 
# --- Project path setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import time
from groq import Groq
from models.keyword_summarizer import KeywordSummarizer
from langsmith.run_helpers import traceable
from rag.foundation_rag import FoundationRAG
from langchain.schema import HumanMessage
from langgraph.graph import END, StateGraph, START
from rag.grade_documents import GradeDocuments
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from states.corrective_rag_state import CorrectiveRAGState
from prompts.query_rewriter_prompt import QUERY_REWRITER_PROMPT
from prompts.grade_documents_prompt import GRADE_DOCUMENTS_PROMPT

class CorrectiveRAG():
    """
    CorrectiveRAG:
    1. A foundational RAG engine (FoundationRAG) for retrieval and generation.
    2. Document grading and filtering to improve answer quality.
    3. Query rewriting when no relevant documents are found.
    4. Web search fallback using Groq if necessary.

    The workflow is implemented as a StateGraph to manage conditional execution.
    """

    def __init__(self, google_api_key, groq_key):
        self.base_rag = FoundationRAG(google_api_key)
        self.groq_key = groq_key
        self.google_api_key = google_api_key
        self.checkpointer = InMemorySaver()
        self.summarizer = KeywordSummarizer()

    
    def get_model(self, state):  
        """
        Add models to the workflow state:
            - filter_model: used for grading document relevance.
            - basic_model: used for query rewriting and generation.
        """
        model_name = "gemini-2.5-flash"

        filter_model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            model_kwargs={"response_mime_type": "application/json"},
            api_key=self.google_api_key
        )

        model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.google_api_key
        )

        state['filter_model'] = filter_model
        state['basic_model'] = model

        return state


    def get_relevant_documents(self, state):
        """Retrieve relevant documents from the vector DB for the current query."""
        query = state["query"]

        documents = self.base_rag.retrieval(query)
        state["relevant_documents"] = documents
        return state
    
    @traceable
    def grade_and_filter_documents(self, state):
        """
        Grade each retrieved document to filter out irrelevant results.
        Uses a structured grading prompt and Pydantic parser.
        """
        filtered_documents = []
        parser = PydanticOutputParser(pydantic_object=GradeDocuments)
        query = state['query']
        model = state['filter_model']
        documents = state['relevant_documents']
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GRADE_DOCUMENTS_PROMPT),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {query}"),
            ]
        )

        retrieval_grader = grade_prompt | model | parser

        for document in documents:
            grader_response = retrieval_grader.invoke({"document": document, "query": query})
            time.sleep(6)
            if grader_response.binary_score.lower() == "yes":
                filtered_documents.append(document)

        state['relevant_documents'] = filtered_documents
        return state

    def generate_answer_from_documents(self, state):
        """
        Generate an answer using the graded documents.
        Uses the base RAG engine to generate response and collects
        metadata from each document, including 'source' and 'university_name'.
        """
        query = state['query']
        documents = state['relevant_documents']
        initial_state = {"messages": [HumanMessage(content=query)]}

        # Generate answer using base RAG
        result = self.base_rag.generation(initial_state, documents)
        state['agent_response'] = result["messages"][-1].content

        # Collect metadata from documents to show the resource 
        if documents:
            sources = [d.metadata.get("source") for d in documents if d.metadata.get("source")]
            universities = [d.metadata.get("university_name") for d in documents if d.metadata.get("university_name")]
            # Remove duplicates and preserve order
            seen = set()
            universities = [u for u in universities if not (u in seen or seen.add(u))]
            metadata = {
                "sources": sources,
                "university_name": universities
            }
        else:
            metadata = None

        state['agent_metadata'] = metadata

        return state

    def decide_generation_source(self,state):
        """Decide whether to generate answer from documents or rewrite query."""
        if len(state['relevant_documents']) > 0:
            return "generate"
        else:
            return "transform_query"
        
    @traceable
    def transform_query(self,state):
        """
        Rewrites the user's query to improve retrieval or generation.
        Uses the basic LLM model with a structured prompt.
        """
        query = state["query"]
        model = state["basic_model"]

        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_REWRITER_PROMPT),
                ("human", "Here is the initial question: \n\n {query} \n Formulate an improved question."),
            ]
        )

        query_rewriter = re_write_prompt | model | StrOutputParser()
        better_query = query_rewriter.invoke({"query": query})
    
        state["query"] = better_query

        return state

    def generate_answer_from_web_search(self,state):
        """
        Fallback generation using web search (Groq) when no documents are found.
        """
        query = state['query']
        short_query = self.summarizer.summarize_text(query, n_phrases=10)

        client = Groq(api_key=self.groq_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": short_query,
                }
            ],
            model="groq/compound", 
        )

        response_text = chat_completion.choices[0].message.content

        state["agent_response"] = (
            "I couldn't get any data from the documents I had, "
            "so I searched the internet and this is what I found:\n\n"
            f"{response_text}"
        )

        state['agent_metadata'] = None
        return state

    def build_graph(self):
        """
        Build and compile the RAG workflow as a StateGraph:
        - Nodes include model setup, retrieval, grading, generation, query transformation, and web search.
        - Conditional edges allow fallback when documents are missing.
        """

        workflow = StateGraph(CorrectiveRAGState)

        # Define the nodes
        workflow.add_node("get_model", self.get_model)
        workflow.add_node("get_relevant_documents", self.get_relevant_documents)
        workflow.add_node("grade_and_filter_documents", self.grade_and_filter_documents)
        workflow.add_node("generate_answer_from_documents", self.generate_answer_from_documents)
        workflow.add_node("generate_answer_from_web_search", self.generate_answer_from_web_search)
        workflow.add_node("transform_query", self.transform_query)

        # Build graph
        workflow.add_edge(START, "get_model")
        workflow.add_edge("get_model", "get_relevant_documents")
        workflow.add_edge("get_relevant_documents", "grade_and_filter_documents")
        workflow.add_conditional_edges(
            "grade_and_filter_documents",
            self.decide_generation_source,
            {
                "transform_query": "transform_query",
                "generate": "generate_answer_from_documents",
            },
        )
        workflow.add_edge("generate_answer_from_documents", END)
        workflow.add_edge("transform_query", "generate_answer_from_web_search")
        workflow.add_edge("generate_answer_from_web_search", END)


        return workflow.compile(checkpointer=self.checkpointer)