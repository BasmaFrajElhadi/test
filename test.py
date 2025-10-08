import os
import sys
import re
import json
from playwright.sync_api import sync_playwright, Page
import os
import sys
import json
import chromadb
from langchain.schema import Document
import re
import json
import os 
import sys
from langdetect import detect
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import MarianMTModel, MarianTokenizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import sqlite3
import json
import os
import sys
import time
import random
from datetime import datetime
from typing import List, Dict, Optional, Any
import time
from groq import Groq
from models.keyword_summarizer import KeywordSummarizer
from langsmith.run_helpers import traceable
from langchain.schema import HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
from langsmith.run_helpers import traceable
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from typing_extensions import TypedDict, NotRequired, Optional
from keybert import KeyBERT
