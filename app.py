
import hashlib
import re
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- LOGIN --------------------

def check_password():
    def password_entered():
        st.session_state["authenticated"] = (
            st.session_state["password_input"]
            == st.secrets["auth"]["password"]
        )

    if st.session_state.get("authenticated", False):
        return True

    st.title("🔐 DocuMind AI Login")
    st.write("Enter your password to access the application.")
    st.text_input(
        "Password",
        type="password",
        key="password_input",
        on_change=password_entered,
    )

    if st.session_state.get("authenticated") is False:
        st.error("Incorrect password. Please try again.")

    return False


if not check_password():
    st.stop()

# -------------------- STYLING --------------------

st.markdown(
    """
    <style>
    .stApp {background:#f8fafc;color:#1e293b}
    [data-testid="stSidebar"] {background:#eaf0f8}
    [data-testid="stSidebar"] * {color:#1e293b}
    .stMarkdown,.stMarkdown p,.stMarkdown li,label,
    [data-testid="stCaptionContainer"] {color:#1e293b}
    .hero {
        padding:30px;
        border-radius:18px;
        background:linear-gradient(135deg,#dbeafe,#e0e7ff);
        border:1px solid #c7d2fe;
        margin-bottom:25px
    }
    .hero h1 {color:#1e3a8a;font-size:38px;margin-bottom:8px}
    .hero p {color:#334155;font-size:16px}
    .feature-card {
        background:#fff;
        border:1px solid #dbe3ef;
        border-radius:12px;
        padding:18px;
        min-height:115px
    }
    .feature-card h4 {color:#3730a3;margin-bottom:8px}
    .feature-card p,.small-muted {color:#475569;font-size:14px}
    div[data-testid="stChatMessage"] {
        background:#fff;
        border:1px solid #dbe3ef;
        border-radius:12px;
        padding:12px
    }
    [data-testid="stChatInput"] textarea {background:#fff;color:#1e293b}
    [data-testid="stFileUploader"] {
        background:#fff;
        border-radius:10px;
        padding:8px
    }
    div.stButton>button {
        background:#fff;
        color:#1e3a8a;
        border-radius:9px;
        border:1px solid #c7d2fe
    }
    div.stButton>button:hover {

        div.stButton>button:hover {
        background:#e0e7ff;
        color:#1e3a8a;
        border-color:#818cf8
    }

    /* Mobile responsive design */
    @media screen and (max-width: 768px) {
        .hero {
            padding: 18px;
            margin-bottom: 16px;
            border-radius: 14px;
        }

        .hero h1 {
            font-size: 27px;
            line-height: 1.25;
        }

        .hero p {
            font-size: 14px;
            line-height: 1.5;
        }

        .feature-card {
            padding: 12px;
            min-height: auto;
            margin-bottom: 10px;
        }

        .feature-card h4 {
            font-size: 16px;
        }

        .feature-card p {
            font-size: 13px;
        }

        div.stButton > button {
            min-height: 44px;
            white-space: normal;
        }

        [data-testid="stChatMessage"] {
            padding: 10px;
            overflow-wrap: anywhere;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
    }
    </style>
        background:#e0e7ff;
        color:#1e3a8a;
        border-color:#818cf8
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- CONFIGURATION --------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-3.6-flash"
FALLBACK_MESSAGE = (
    "I couldn't find that information in the uploaded documents."
)

# -------------------- LOCAL EMBEDDINGS --------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# -------------------- GEMINI CHAT MODEL --------------------

@st.cache_resource
def get_chat_model():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
        google_api_key=st.secrets["GOOGLE_API_KEY"],
    )


# -------------------- SESSION STATE --------------------

DEFAULTS = {
    "messages": [],
    "vector_store": None,
    "collection_hash": None,
    "total_pages": 0,
    "total_chunks": 0,
    "uploaded_names": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------- SOURCE CITATIONS --------------------

def show_sources(sources):
    if sources:
        with st.expander("📎 View source citations"):
            for item in sources:
                st.write(
                    f"📄 {item.get('source', 'Unknown PDF')} "
                    f"— Page {item.get('page', 'Unknown')}"
                )


# -------------------- ANSWER CLEANING --------------------

def clean_answer(text):
    text = str(text or "").strip()
    text = re.sub(r"(?m)^[ \t]*[-*][ \t]*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_response_text(response):
    content = response.content

    if isinstance(content, str):
        return clean_answer(content)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)

        return clean_answer("\n".join(parts))

    return clean_answer(str(content))


# -------------------- PDF INDEXING --------------------

def build_vector_store(uploaded_files):
    documents = []
    file_names = []
    total_pages = 0
    hasher = hashlib.sha256()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        add_start_index=True,
    )

    for uploaded_file in uploaded_files:
        raw = uploaded_file.getvalue()
        name = uploaded_file.name

        file_names.append(name)
        hasher.update(name.encode("utf-8"))
        hasher.update(raw)

        try:
            reader = PdfReader(BytesIO(raw))

            if reader.is_encrypted:
                raise ValueError("This PDF is password-protected.")

            total_pages += len(reader.pages)

            for page_no, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""

                if page_text.strip():
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": name,
                                "page": page_no,
                            },
                        )
                    )

        except Exception as error:
            st.warning(f"Could not read {name}: {error}")

    collection_hash = hasher.hexdigest()

    if not documents:
        return None, collection_hash, total_pages, 0, file_names

    chunks = splitter.split_documents(documents)

    if not chunks:
        return None, collection_hash, total_pages, 0, file_names

    # Embeddings are generated locally; this step does not use Gemini embeddings.
    vector_store = FAISS.from_documents(
        chunks,
        get_embeddings(),
    )

    return (
        vector_store,
        collection_hash,
        total_pages,
        len(chunks),
        file_names,
    )


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## 📚 DocuMind AI")
    st.caption("Your personal PDF assistant")
    st.divider()

    st.markdown("### 📄 Upload documents")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDFs and ask questions about their contents.",
    )

    st.divider()
    st.markdown("### 📊 Document statistics")
    st.metric("PDF files", len(st.session_state.uploaded_names))
    st.metric("Total pages", st.session_state.total_pages)
    st.metric("Text chunks", st.session_state.total_chunks)

    if st.session_state.uploaded_names:
        with st.expander("View uploaded files"):
            for name in st.session_state.uploaded_names:
                st.write(f"📄 {name}")

    st.divider()

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### ⚙️ Configuration")
    st.caption(f"Chat model: {CHAT_MODEL}")
    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    st.caption("Vector database: FAISS")
    st.caption("Embeddings: Local HuggingFace model")
    st.caption("Chat generation: Google Gemini API")

    st.divider()
    st.markdown(
        '<p class="small-muted">'
        'PDF embeddings run locally. Answer generation uses Gemini API.'
        '</p>',
        unsafe_allow_html=True,
    )


# -------------------- MAIN PAGE --------------------

st.markdown(
    """
    <div class="hero">
      <h1>📚 DocuMind AI</h1>
      <p>Chat with your PDF documents using Retrieval-Augmented Generation
      (RAG), semantic search and conversation memory.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(3)

cards = [
    ("📄 Document Search", "Retrieve relevant passages from your PDF files."),
    ("🧠 Conversation Memory", "Ask follow-up questions using earlier conversation context."),
    ("🔎 Source Citations", "Check which PDF and page support the answer."),
]

for col, (heading, body) in zip(cols, cards):
    with col:
        st.markdown(
            f'<div class="feature-card"><h4>{heading}</h4>'
            f'<p>{body}</p></div>',
            unsafe_allow_html=True,
        )

st.write("")

# -------------------- BUILD OR UPDATE INDEX --------------------

if uploaded_files:
    current_hasher = hashlib.sha256()

    for file in uploaded_files:
        current_hasher.update(file.name.encode("utf-8"))
        current_hasher.update(file.getvalue())

    current_hash = current_hasher.hexdigest()

    if current_hash != st.session_state.collection_hash:
        with st.spinner(
            "Reading PDFs and building the local search index..."
        ):
            try:
                store, new_hash, pages, chunks, names = build_vector_store(
                    uploaded_files
                )

                st.session_state.vector_store = store
                st.session_state.collection_hash = new_hash
                st.session_state.total_pages = pages
                st.session_state.total_chunks = chunks
                st.session_state.uploaded_names = names
                st.session_state.messages = []

                if store is not None:
                    st.success(
                        f"Indexed {chunks} text chunks from "
                        f"{len(names)} PDF file(s)."
                    )
                else:
                    st.warning(
                        "No readable text was found in the uploaded PDFs."
                    )

            except Exception as error:
                st.session_state.vector_store = None
                st.error(
                    "Could not build the document index. "
                    "Check the local embedding model installation and "
                    f"available memory. Details: {error}"
                )

else:
    if st.session_state.collection_hash is not None:
        st.session_state.vector_store = None
        st.session_state.collection_hash = None
        st.session_state.total_pages = 0
        st.session_state.total_chunks = 0
        st.session_state.uploaded_names = []
        st.session_state.messages = []


# -------------------- SUGGESTED QUESTIONS --------------------

selected_question = None

if st.session_state.vector_store is not None:
    st.markdown("### 💡 Suggested questions")

    suggestion_cols = st.columns(3)
    suggestions = [
        "Summarize the document",
        "What are the main topics?",
        "Give me important points",
    ]

    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i]:
            if st.button(
                suggestion,
                use_container_width=True,
                key=f"suggestion_{i}",
            ):
                selected_question = suggestion

    st.write("")


# -------------------- CHAT HISTORY --------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(message.get("sources", []))


# -------------------- QUESTION INPUT --------------------

if st.session_state.vector_store is None:
    st.info(
        "👈 Upload at least one readable PDF from the sidebar "
        "to start chatting with your documents."
    )
    question = None
else:
    typed_question = st.chat_input(
        "Ask a question about your documents..."
    )
    question = typed_question or selected_question


# -------------------- QUESTION ANSWERING --------------------

if question and question.strip():
    question = question.strip()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("assistant"):
        with st.spinner("Searching your PDFs and preparing an answer..."):
            try:
                scored_docs = (
                    st.session_state.vector_store
                    .similarity_search_with_score(question, k=6)
                )

                retrieved_docs = [doc for doc, _score in scored_docs]

                if not retrieved_docs:
                    answer, sources = FALLBACK_MESSAGE, []

                else:
                    context_parts = []
                    sources = []
                    seen = set()

                    for doc in retrieved_docs:
                        source = doc.metadata.get("source", "Unknown PDF")
                        page = doc.metadata.get("page", "Unknown")

                        context_parts.append(
                            f"[Source: {source}, Page: {page}]\n"
                            f"{doc.page_content}"
                        )

                        source_key = (source, page)

                        if source_key not in seen:
                            sources.append(
                                {"source": source, "page": page}
                            )
                            seen.add(source_key)

                    context = "\n\n---\n\n".join(context_parts)

                    history = []

                    for item in st.session_state.messages[:-1]:
                        if item["role"] == "user":
                            history.append(f"User: {item['content']}")
                        elif item["role"] == "assistant":
                            history.append(
                                f"Assistant: {item['content']}"
                            )

                    history_text = (
                        "\n".join(history) or "No previous conversation."
                    )

                    prompt = ChatPromptTemplate.from_template(
                        """
You are DocuMind AI, a PDF question-answering assistant.

Answer the CURRENT QUESTION using only facts directly supported by
the supplied PDF passages. Do not use general knowledge or facts
from model training.

Instructions:
- Identify the subject and intent of the current question first.
- Use only passages relevant to the current question.
- Ignore passages about unrelated topics.
- Conversation history is only for resolving follow-up references,
  never evidence.
- If the passages do not directly support an answer, reply exactly:
  "I couldn't find that information in the uploaded documents."
- If only some requested facts are supported, state only those facts.
- Do not add unsupported facts, examples, or definitions.
- Do not include citation labels or empty bullet lists in the answer.
- Keep the answer clear and concise.

Conversation history (context only):
{history}

Retrieved PDF passages (possible evidence):
{context}

Current question:
{question}

Answer:
"""
                    )

                    response = (prompt | get_chat_model()).invoke(
                        {
                            "history": history_text,
                            "context": context,
                            "question": question,
                        }
                    )

                    answer = extract_response_text(response)

                    if not answer or answer == FALLBACK_MESSAGE:
                        answer = FALLBACK_MESSAGE
                        sources = []
                    else:
                        answer = clean_answer(answer)

                st.markdown(answer)
                show_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                error_message = (
                    "Sorry, I couldn't generate an answer. "
                    "Check GOOGLE_API_KEY in Streamlit Secrets and "
                    "your Gemini chat API quota. Details: "
                    f"{error}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
