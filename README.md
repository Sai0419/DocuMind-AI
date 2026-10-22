# DocuMind AI — Chat with Your PDFs

An AI-powered PDF question-answering application built using Python, Streamlit, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs).

DocuMind AI allows users to upload PDF documents and ask questions about their contents. It retrieves relevant passages from the uploaded documents and uses a language model to generate context-based answers.

## 🚀 Features

- **Multiple PDF Uploads:** Upload multiple PDF documents for analysis.
- **Document Question Answering:** Ask questions using natural language.
- **RAG Pipeline:** Retrieve relevant document content before generating answers.
- **Text Chunking:** Split extracted PDF text into smaller, manageable chunks.
- **Semantic Search:** Generate embeddings and retrieve relevant passages using FAISS.
- **Local LLM Integration:** Generate responses using Ollama.
- **Source References:** View supporting document passages and page references.
- **Chat Interface:** Interact with your documents through a Streamlit interface.
- **Conversation History:** Maintain the conversation during a session.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | Document processing and LLM integration |
| PyPDF | Extract text from PDF documents |
| Ollama | Run language models locally |
| Qwen 2.5 (1.5B) | Answer generation |
| Nomic Embed Text | Text embeddings |
| FAISS | Vector similarity search |

## 🧠 How It Works

1. **Upload PDFs:** The user uploads one or more PDF documents.
2. **Extract Text:** Text is extracted from the uploaded PDFs.
3. **Split Text:** The extracted text is divided into smaller chunks with overlapping content.
4. **Generate Embeddings:** Each chunk is converted into a numerical vector using the embedding model.
5. **Store and Retrieve:** FAISS searches for document chunks relevant to the user's question.
6. **Generate an Answer:** The retrieved context is passed to the local language model.
7. **Display Results:** The application presents the answer and supporting document references.

### Architecture

`PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Retrieval → Ollama LLM → Answer with Sources`

## 📋 Prerequisites

Before running the application, install:

- Python 3.10 or a compatible version
- Git
- Ollama
- The required Ollama language and embedding models

Pull the models using:

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the application.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sai0419/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Git Bash:**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

`http://localhost:8501`

Open that URL in your browser.

## 💬 Example Use Cases

- Summarize information from a PDF.
- Find specific information in lengthy documents.
- Ask questions about study materials.
- Retrieve relevant passages from technical documentation.
- Explore reports and other text-based PDF files.

Answers depend on the text extracted from the uploaded documents and the quality of the retrieved context.

## 📁 Project Structure

```text
DocuMind-AI/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔒 Privacy and Limitations

- The language model and embedding model run locally through Ollama.
- Uploaded PDFs are processed by the application for retrieval and question answering.
- Answer quality depends on PDF text extraction, retrieval relevance, and model performance.
- Scanned PDFs may require OCR if they do not contain extractable text.
- The application requires sufficient system resources to run the selected models.

## 🔮 Future Improvements

- Add PDF summarization and conversation export.
- Improve retrieval using reranking and metadata filters.
- Add automated tests and error handling.
- Add support for more document formats.
- Deploy a hosted version with appropriate model-serving infrastructure.

## 👨‍💻 Author

**GS**

GitHub: [@Sai0419](https://github.com/Sai0419)

## 📄 License

A license has not yet been specified. Add a suitable open-source license before inviting others to reuse or distribute this project.
