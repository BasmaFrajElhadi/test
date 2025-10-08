![Alt text](Downloads\Egyptian Universities Corrective RAG App.svg)

# 🏛️ Egyptian Universities Corrective RAG App

*A Corrective Retrieval-Augmented Generation (RAG) Application for Public Egyptian Universities*

---

## 📘 Overview

The **Egyptian Universities RAG App** is an intelligent **Retrieval-Augmented Generation (RAG)** system designed to answer questions about **public universities in Egypt**.
It automatically scrapes, processes, and structures university information, then leverages **Large Language Models (LLMs)** to provide **accurate, explainable, and document-grounded** answers to user queries through an **interactive Streamlit interface**.

---

## 🎯 Project Objectives

* Create an intelligent Q&A system about Egyptian public universities.
* Automate data collection, cleaning, translation, and embedding.
* Build a vector-based retrieval system integrated with an LLM for grounded responses.
* Provide an easy-to-use **Streamlit web interface** for students and researchers.
* Deploy the complete system on **Streamlit Cloud** for public access.

---

## 🧩 System Architecture

The project consists of **three main layers**:

### 1. **Data Layer**

* **Scraping:** Using Playwright to scrape data from [universitiesegypt.com](https://www.universitiesegypt.com/).
* **Cleaning & Translation:** Removing HTML, punctuation, and translating Arabic to English.
* **Flattening & Chunking:** Converting structured data into text documents and splitting them for embeddings.
* **Storage:** Saving documents and metadata into a **ChromaDB** vector store for semantic retrieval.

### 2. **RAG Engine**

The **RAG Engine** powers the intelligent retrieval and generation workflow of the assistant.
It’s implemented in the `CorrectiveRAG` class and integrates **Gemini 2.5 Flash** and **LangSmith** monitoring to ensure accuracy and reliability.

* **Retriever:**
  Retrieves the most relevant document chunks from the vector database using the `FoundationRAG` module.

* **LLM Integration:**
  Uses **Google Gemini 2.5 Flash** via `ChatGoogleGenerativeAI` for both:

  * **Document Grading & Filtering:** Evaluates retrieved content relevance using structured prompts and `PydanticOutputParser`.
  * **Answer Generation:** Produces context-aware, grounded answers using the filtered knowledge.

* **Pipeline:**

  ```
  User Query → Query Rewrite → Retriever → Document Grader → LLM → Final Answer
  ```

* **Fallback Search (Groq):**
  When no relevant results are found in the local knowledge base, the system automatically performs a web search using the **Groq API**, ensuring continuity in responses.

* **Monitoring (LangSmith):**
  Each step — from query rewriting to document grading and generation — is **traced and monitored** using **LangSmith**, enabling:

  * Performance tracking and latency measurement
  * Retrieval quality evaluation
  * Debugging of model prompts and responses

### 3. **Application Layer (Streamlit GUI)**

* **Ask Questions:** Users can type questions about universities.
* **Show Answers:** Generated answers displayed with source.
* **Recent Questions:** Keeps history of past queries.
* **Deployment:** Hosted on **Streamlit Cloud** for accessibility.

---

## ⚙️ Functional Workflow

1. **Data Collection**

   * Scrape all public universities.
   * Extract about pages, faculties, contact info, and admission details.

2. **Data Processing**

   * Clean text (remove tags, whitespace, etc.)
   * Translate Arabic → English.
   * Flatten JSON into text + metadata.
   * Chunk into segments (1000 characters, 100 overlap).

3. **Embedding & Storage**

   * Generate sentence embeddings using **`all-MiniLM-L6-v2`**.
   * Store chunks and metadata in **ChromaDB**.

4. **Retrieval-Augmented Generation**

   * Query reformulation → top-k retrieval → LLM response.
   * Return grounded answers with cited sources.

5. **User Interface**

   * Streamlit app for question-answering, document inspection, and session tracking.

---

## 🧠 Core Components

| Component             | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| **WebScraper**        | Extracts data from `universitiesegypt.com` using Playwright.   |
| **TextProcessor**     | Cleans, translates, flattens, and chunks raw text data.        |
| **TextEmbedder**      | Generates embeddings for chunks using `SentenceTransformer`.   |
| **VectorDB (Chroma)** | Stores embeddings and metadata for fast semantic retrieval.    |
| **FoundationRAG**     | Core RAG logic: retrieval, context augmentation, generation.   |
| **SQLiteChatStorage** | Stores session chats, metadata, and user histories.            |
| **Streamlit App**     | User-facing GUI for querying, viewing, and managing responses. |

---

## 🛠️ Technologies Used

| Category            | Technology                                              |
| ------------------- | ------------------------------------------------------- |
| **Language**        | Python 3.10+                                            |
| **Scraping**        | Playwright                                              |
| **Data Processing** | MarianMT (Helsinki-NLP/opus-mt-ar-en), Regex, LangChain |
| **Vector Database** | ChromaDB                                                |
| **Embeddings**      | Sentence Transformers (MiniLM-L6-v2)                    |
| **LLM Integration** | Gemini, Llama 3, or GPT-4                               |
| **Frontend**        | Streamlit                                               |
| **Database**        | SQLite                                                  |
| **PDF Reports**     | ReportLab                                               |
| **Deployment**      | Streamlit Cloud                                         |

---

## 🧰 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/BasmaFrajElhadi/egyptian-universities-corrective-rag.git
cd egyptian-universities-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Example Queries

* “What is the admission requirement for Alexandria University?”
* “List the faculties available at Mansoura University.”
* “How can I contact Cairo University?”
* “Which universities have research centers?”
* “What is the rating of Ain Shams University?”

---

## 📜 License

This project is licensed under the **MIT License** — you’re free to use, modify, and distribute it with attribution.

