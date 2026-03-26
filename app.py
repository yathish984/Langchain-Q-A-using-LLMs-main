import os
import asyncio
import nest_asyncio
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader

# ----------------------
# Fix for async + Streamlit
# ----------------------
nest_asyncio.apply()
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---------------------------
# Page config + styling
# ---------------------------
st.set_page_config(page_title="📖 Multi-Source QA App", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.main .block-container {background-color: #eef2f7; padding: 2rem; color: #2c3e50;}
.header-container {background-color: #e0f7fa; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; border: 2px solid #b2ebf2;}
.header-container h1 {color: #00796b; font-size: 2.5em; font-weight: bold;}
.content-box {background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0f2f1; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
.stButton>button {background-color: #00796b; color: white; font-size: 16px; padding: 10px 20px; border-radius: 8px; border: none; transition: background-color 0.3s;}
.stButton>button:hover {background-color: #004d40;}
.stTextInput>div>div>input {border-radius: 8px; border: 1px solid #ccc; padding: 8px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# App state
# ---------------------------
if "page" not in st.session_state: st.session_state.page = "Home"
if "history" not in st.session_state: st.session_state.history = []

def set_page(page_name): st.session_state.page = page_name

# ---------------------------
# Header + Navigation
# ---------------------------
st.markdown('<div class="header-container"><h1>Langchain Q&A using Gemini 🧑‍🏫</h1></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1: 
    if st.button("Home"): set_page("Home")
with col2: 
    if st.button("Process URL"): set_page("URL")
with col3: 
    if st.button("Process PDF"): set_page("PDF")

# ---------------------------
# API Key
# ---------------------------
api_key = st.text_input("🔑 Enter your Google API key:", type="password")
if api_key: os.environ["GOOGLE_API_KEY"] = api_key

# ---------------------------
# Utility Functions
# ---------------------------
@st.cache_data
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

@st.cache_data
def extract_text_from_url(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        st.error(f"Failed to load URL: {e}")
        return ""

@st.cache_data
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def ask_gemini(model_name, context, question, languages):
    language_list = ", ".join(languages)
    prompt = f"""
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say "Answer is not available in the context."
The answer must be provided in all of the following languages: {language_list}.

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
    return model.invoke(prompt)

# ---------------------------
# Main Logic
# ---------------------------
def main():
    if st.session_state.page == "Home":
        st.markdown("""
        <div class="content-box">
        <h2>Welcome to the Multi-Source Q&A App!</h2>
        <p>Use the tabs to process PDF files or URLs and ask questions directly from the content.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<div class="content-box">', unsafe_allow_html=True)

    language_map = {
        "English":"en-US", "Spanish":"es-US", "French":"fr-FR",
        "German":"de-DE","Japanese":"ja-JP","Hindi":"hi-IN",
        "Kannada":"kn-IN","Telugu":"te-IN"
    }

    source_input = None
    user_question = None

    if st.session_state.page == "PDF":
        source_input = st.file_uploader("📂 Upload a PDF file:", type="pdf")
        user_question = st.text_area("❓ Enter your question:", height=100)
        selected_languages = st.multiselect("Select answer languages:", options=list(language_map.keys()), default=["English"])
    elif st.session_state.page == "URL":
        source_input = st.text_input("🔗 Enter URL:", placeholder="https://example.com")
        user_question = st.text_area("❓ Enter your question:", height=100)
        selected_languages = st.multiselect("Select answer languages:", options=list(language_map.keys()), default=["English"])

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Get Answer"):
        if not api_key: st.warning("Please enter your Google API key.")
        elif not user_question: st.warning("Please enter a question.")
        elif st.session_state.page=="PDF" and not source_input: st.warning("Please upload a PDF.")
        elif st.session_state.page=="URL" and not source_input: st.warning("Please enter a URL.")
        else:
            try:
                with st.spinner("Processing content..."):
                    if st.session_state.page=="PDF":
                        raw_text = extract_text_from_pdf(source_input)
                    else:
                        raw_text = extract_text_from_url(source_input)

                    if not raw_text:
                        st.error("No text could be extracted.")
                        return

                    chunks = get_text_chunks(raw_text)
                    context = "\n".join(chunks[:10])  # limit to avoid token overflow

                    with st.spinner("Generating answer..."):
                        try:
                            response = ask_gemini("gemini-2.5-flash", context, user_question, selected_languages)
                        except Exception as e:
                            st.error(f"⚠️ Error generating answer: {e}")
                            return

                st.session_state.history.append({"question": user_question, "answer": response.content})

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")

    # Show history
    if st.session_state.history:
        st.subheader("💬 Conversation History")
        for i, chat in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Q{len(st.session_state.history)-i}:** {chat['question']}")
            st.info(chat['answer'])

        if st.button("🗑️ Clear Chat"):
            st.session_state.history = []
            st.experimental_rerun()