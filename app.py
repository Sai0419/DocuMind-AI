
import hashlib
from io import BytesIO

import streamlit as st
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# PASSWORD PROTECTION
# --------------------------------------------------

def check_password():
    def password_entered():
        entered_password = st.session_state["password_input"]
        expected_password = st.secrets["auth"]["password"]

        if entered_password == expected_password:
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False

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


# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------


# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #eaf0f8;
    }

    [data-testid="stSidebar"] * {
        color: #1e293b;
    }

    /* Main text */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    label,
    [data-testid="stCaptionContainer"] {
        color: #1e293b;
    }

    /* Hero banner */
    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #dbeafe, #e0e7ff);
        border: 1px solid #c7d2fe;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #1e3a8a;
        font-size: 38px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #334155;
        font-size: 16px;
    }

    /* Feature cards */
    .feature-card {
        background-color: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        padding: 18px;
        min-height: 115px;
    }

    .feature-card h4 {
        color: #3730a3;
        margin-bottom: 8px;
    }

    .feature-card p {
        color: #475569;
        font-size: 14px;
    }

    .small-muted {
        color: #475569;
        font-size: 13px;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        padding: 12px;
    }

    /* Input fields */
    [data-testid="stChatInput"] textarea {
        background-color: #ffffff;
        color: #1e293b;
    }

    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 8px;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #ffffff;
        color: #1e3a8a;
        border-radius: 9px;
        border: 1px solid #c7d2fe;
    }

    div.stButton > button:hover {
        background-color: #e0e7ff;
        color: #1e3a8a;
        border-color: #818cf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




# --------------------------------------------------
# 3. OLLAMA MODELS
# --------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "qwen2.5:1.5b"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_chat_model():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
    )


# --------------------------------------------------
# 4. SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "collection_hash" not in st.session_state:
    st.session_state.collection_hash = None

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []


# --------------------------------------------------
# 5. PDF PROCESSING AND VECTOR INDEX
# --------------------------------------------------

def build_vector_store(uploaded_files):
    documents = []
    file_names = []
    total_pages = 0
    collection_hasher = hashlib.sha256()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        add_start_index=True,
    )

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        file_names.append(file_name)

        # Include both file name and contents in the hash.
        collection_hasher.update(file_name.encode("utf-8"))
        collection_hasher.update(file_bytes)

        try:
            pdf_reader = PdfReader(BytesIO(file_bytes))

            if pdf_reader.is_encrypted:
                raise ValueError(
                    "This PDF is password-protected."
                )

            for page_number, page in enumerate(
                pdf_reader.pages, start=1
            ):
                page_text = page.extract_text() or ""

                if not page_text.strip():
                    continue

                from langchain_core.documents import Document

                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": file_name,
                            "page": page_number,
                        },
                    )
                )

            total_pages += len(pdf_reader.pages)

        except Exception as error:
            st.warning(
                f"Could not read {file_name}: {error}"
            )

    collection_hash = collection_hasher.hexdigest()

    if not documents:
        return None, collection_hash, total_pages, 0, file_names

    chunks = splitter.split_documents(documents)

    if not chunks:
        return None, collection_hash, total_pages, 0, file_names

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return (
        vector_store,
        collection_hash,
        total_pages,
        len(chunks),
        file_names,
    )


# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------

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

    st.metric(
        "PDF files",
        len(st.session_state.uploaded_names),
    )

    st.metric(
        "Total pages",
        st.session_state.total_pages,
    )

    st.metric(
        "Text chunks",
        st.session_state.total_chunks,
    )

    if st.session_state.uploaded_names:
        with st.expander("View uploaded files"):
            for name in st.session_state.uploaded_names:
                st.write(f"📄 {name}")

    st.divider()

    if st.button(
        "🗑️ Clear chat history",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### ⚙️ Configuration")
    st.caption(f"Chat model: {CHAT_MODEL}")
    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    st.caption("Vector database: FAISS")

    st.divider()

    st.markdown(
        """
        <p class="small-muted">
        Runs locally with Python, Streamlit and Ollama.
        </p>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 7. MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📚 DocuMind AI</h1>
        <p>
        Chat with your PDF documents using Retrieval-Augmented
        Generation (RAG), semantic search and conversation memory.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 8. FEATURE CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <h4>📄 Document Search</h4>
            <p>Retrieve relevant passages from your PDF files.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <h4>🧠 Conversation Memory</h4>
            <p>Ask follow-up questions using earlier conversation context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <h4>🔎 Source Citations</h4>
            <p>Check which PDF and page support the answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# --------------------------------------------------
# 9. PROCESS UPLOADED PDF FILES
# --------------------------------------------------

if uploaded_files:
    current_hasher = hashlib.sha256()

    for uploaded_file in uploaded_files:
        current_hasher.update(
            uploaded_file.name.encode("utf-8")
        )
        current_hasher.update(uploaded_file.getvalue())

    current_hash = current_hasher.hexdigest()

    if current_hash != st.session_state.collection_hash:
        with st.spinner(
            "Reading PDFs and building the search index..."
        ):
            try:
                (
                    new_vector_store,
                    new_hash,
                    total_pages,
                    total_chunks,
                    file_names,
                ) = build_vector_store(uploaded_files)

                st.session_state.vector_store = new_vector_store
                st.session_state.collection_hash = current_hash
                st.session_state.total_pages = total_pages
                st.session_state.total_chunks = total_chunks
                st.session_state.uploaded_names = file_names

                # A new document collection starts a fresh chat.
                st.session_state.messages = []

                if new_vector_store is not None:
                    st.success(
                        f"Indexed {total_chunks} text chunks "
                        f"from {len(file_names)} PDF file(s)."
                    )
                else:
                    st.warning(
                        "No readable text was found in the uploaded PDFs."
                    )

            except Exception as error:
                st.error(
                    "Could not build the document index. "
                    "Check that Ollama is running and the embedding "
                    f"model is installed. Details: {error}"
                )
else:
    if st.session_state.collection_hash is not None:
        st.session_state.vector_store = None
        st.session_state.collection_hash = None
        st.session_state.total_pages = 0
        st.session_state.total_chunks = 0
        st.session_state.uploaded_names = []
        st.session_state.messages = []


# --------------------------------------------------
# 10. SUGGESTED QUESTIONS
# --------------------------------------------------

if st.session_state.vector_store is not None:
    st.markdown("### 💡 Suggested questions")

    suggestion_cols = st.columns(3)

    suggestions = [
        "Summarize the document",
        "What are the main topics?",
        "Give me important points",
    ]

    selected_question = None

    for index, suggestion in enumerate(suggestions):
        with suggestion_cols[index]:
            if st.button(
                suggestion,
                use_container_width=True,
                key=f"suggestion_{index}",
            ):
                selected_question = suggestion

    st.write("")


# --------------------------------------------------
# 11. DISPLAY PREVIOUS CHAT MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("📎 View source citations"):
                    for source in sources:
                        st.markdown(
                            f"- **{source['source']}** "
                            f"— Page {source['page']}"
                        )


# --------------------------------------------------
# 12. CHAT INPUT
# --------------------------------------------------

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


# --------------------------------------------------
# 13. RETRIEVAL + CONVERSATION MEMORY + ANSWERING
# --------------------------------------------------

if question:
    question = question.strip()

    if question:
        # Save the current question.
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching your PDFs and preparing an answer..."
            ):
                try:
                    # Step A: Retrieve relevant document chunks.
                    relevant_docs = (
                        st.session_state.vector_store
                        .similarity_search(question, k=6)
)

                    # Step B: Build document context and source list.
                    context_parts = []
                    sources = []
                    seen_sources = set()

                    for document in relevant_docs:
                        source_name = document.metadata.get(
                            "source", "Unknown PDF"
                        )
                        page_number = document.metadata.get(
                            "page", "Unknown"
                        )

                        context_parts.append(
                            f"[Source: {source_name}, "
                            f"Page: {page_number}]\n"
                            f"{document.page_content}"
                        )

                        source_key = (
                            source_name,
                            page_number,
                        )

                        if source_key not in seen_sources:
                            sources.append(
                                {
                                    "source": source_name,
                                    "page": page_number,
                                }
                            )
                            seen_sources.add(source_key)

                    context = "\n\n---\n\n".join(context_parts)

                    # Step C: Prepare conversation history.
                    # Exclude the current question, which was just appended.
                    chat_history = []

                    for message in st.session_state.messages[:-1]:
                        if message["role"] == "user":
                            chat_history.append(
                                f"User: {message['content']}"
                            )
                        elif message["role"] == "assistant":
                            chat_history.append(
                                f"Assistant: {message['content']}"
                            )

                    history_text = "\n".join(chat_history)

                    # Step D: Prompt the model with history + retrieved text.
                    prompt = ChatPromptTemplate.from_template(
                        """
You are DocuMind AI, a helpful document question-answering assistant.

Use the conversation history to understand follow-up questions.
Use the relevant document passages as the source of factual answers.

Rules:
1. Base factual answers on the provided document passages.
2. Use conversation history to understand questions like
   "What are its advantages?" or "Explain that again".
3. Previous AI answers are context, not proof that a fact is true.
4. If the answer cannot be found in the passages, say:
   "I couldn't find that information in the uploaded documents."
5. Answer clearly in simple language.
6. Do not invent facts or source citations.
7. If a question is ambiguous, use the conversation history
   to interpret it when possible.

Previous conversation:
{history}

Relevant document passages:
{context}

Current question:
{question}

Answer:
"""
                    )

                    chain = prompt | get_chat_model()

                    response = chain.invoke(
                        {
                            "history": (
                                history_text
                                if history_text
                                else "No previous conversation."
                            ),
                            "context": context,
                            "question": question,
                        }
                    )

                    answer = response.content

                    # Display the answer.
                    st.markdown(answer)

                    # Display sources for this answer.
                    if sources:
                        with st.expander("📎 View source citations"):
                            for source in sources:
                                st.markdown(
                                    f"- **{source['source']}** "
                                    f"— Page {source['page']}"
                                )

                    # Save answer and citations for future turns.
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
                        "Check that Ollama is running and that both "
                        f"models are installed. Details: {error}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )
