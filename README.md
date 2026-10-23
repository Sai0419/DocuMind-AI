# DocuMind AI — Chat with Your PDFs

DocuMind AI is a local AI-powered PDF chatbot that allows users to upload documents and ask questions in natural language. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from PDF files and generate answers with source-page references.

The project combines Python, Streamlit, LangChain, FAISS, and Ollama to provide a document question-answering experience without relying on paid cloud AI APIs.

## Application Preview

<!-- Add a screenshot of your working application -->
<!-- Save it as screenshots/documind-preview.png -->

![DocuMind AI Application Preview](screenshots/documind-ai.png.png)

## Features

- Upload and process PDF documents.
- Ask questions about document content in natural language.
- Retrieve relevant text using semantic search.
- Generate context-based answers using a local language model.
- Display source-page references for answer verification.
- Support multiple PDF uploads.
- Provide a password-protected login page.
- Run locally using free, open-source tools.
- Avoid paid AI API dependencies in the core workflow.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | LLM and retrieval integration |
| Ollama | Runs local AI models |
| Qwen 2.5 1.5B | Local chat model |
| Nomic Embed Text | Text embedding generation |
| FAISS | Vector storage and similarity search |
| PyPDF | PDF text extraction |

## How It Works

1. **PDF Upload:** The user uploads one or more PDF files.
2. **Text Extraction:** Text is extracted from the uploaded documents.
3. **Text Chunking:** The extracted text is divided into smaller chunks.
4. **Embedding Generation:** The embedding model converts text chunks into vectors.
5. **Vector Search:** FAISS retrieves chunks relevant to the user's question.
6. **Answer Generation:** The retrieved content is passed to the local language model.
7. **Source References:** The application displays an answer with source-page references where available.

## Architecture

```text
              User
               |
               v
       Streamlit Interface
               |
               v
          PDF Upload
               |
               v
       PDF Text Extraction
               |
               v
          Text Chunking
               |
               v
       Nomic Embeddings
               |
               v
       FAISS Vector Search
               |
               v
      Relevant Text Chunks
               |
               v
      Qwen 2.5 via Ollama
               |
               v
     Answer + Page References
```

## Prerequisites

Before running the application, install:

- Python compatible with the project's dependencies
- Git
- Ollama
- The local AI models listed below
- Sufficient memory and disk space for the models

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sai0419/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create a Virtual Environment

**Windows — PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and Download Ollama Models

Install Ollama from [ollama.com](https://ollama.com/), start it, and download the models:

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

These model names should match the configuration in `rag_pipeline.py`.

### 5. Configure the Application Password

If the application uses Streamlit secrets for authentication, create the following file in the project root:

`.streamlit/secrets.toml`

Add:

```toml
[auth]
password = "YOUR_STRONG_PASSWORD"
```

Replace the placeholder with your own strong password. Do not commit this file or publish the password.

### 6. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal. By default, Streamlit commonly uses:

`http://localhost:8501`

## Example Use Cases

- Ask questions about Python interview preparation PDFs.
- Find specific information in technical documentation.
- Review educational materials and study notes.
- Retrieve relevant details from lengthy documents.
- Ask questions about uploaded reports and reference materials.

## Project Structure

```text
DocuMind-AI/
├── .streamlit/
│   └── secrets.toml       # Local secrets; do not commit
├── app.py                 # Streamlit application interface
├── rag_pipeline.py        # RAG and retrieval pipeline
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Excluded files and secrets
└── screenshots/
    └── documind-preview.png
```

The screenshot and secrets paths shown above should exist locally as appropriate. The secrets file must remain excluded from Git.

## Live Demo

**Live Demo:** Add your active demo URL here.

The demo may use a temporary public tunnel connected to the local application. In that case, the computer, application, internet connection, and tunnel must remain active. The URL may change when the tunnel restarts.

For security, do not upload confidential PDFs or expose local services such as the Ollama API directly to the internet.

## Source Code

**GitHub Repository:**  
https://github.com/Sai0419/DocuMind-AI

## Privacy and Limitations

### Privacy

- The application is designed to use locally running AI models instead of paid cloud AI APIs.
- PDF processing and answer generation are intended to run locally.
- Uploaded documents may be handled in memory or stored temporarily, depending on the implementation.
- Passwords and configuration secrets should never be committed to the repository.
- When exposing the application through a public tunnel, use strong authentication and avoid sensitive documents.

### Limitations

- Answer quality depends on the content and readability of the uploaded PDFs.
- Scanned PDFs may need OCR support if they contain no extractable text.
- Local model performance depends on available memory and processing power.
- Generated answers can be inaccurate or incomplete and should be verified against the original documents.
- Source-page references help locate relevant information but do not guarantee correctness.
- A temporary tunnel does not provide permanent hosting or guaranteed availability.

## Future Improvements

- Improve retrieval quality and answer accuracy.
- Add conversation history and document management.
- Support additional document formats, such as DOCX and TXT.
- Add OCR for scanned PDFs.
- Improve performance and error handling.
- Add more robust authentication and access controls.
- Deploy a reliable hosted version with suitable privacy and security protections.

## Author

**GS**

- GitHub Profile: https://github.com/Sai0419
- Project Repository: https://github.com/Sai0419/DocuMind-AI

## License

This project does not specify a license yet.

If you want others to reuse, modify, and distribute your code, you can choose an open-source license such as the MIT License. Add a `LICENSE` file with the full license text before claiming that the project is MIT-licensed.
