import os
import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

st.set_page_config(page_title="DocuMind AI | PDF Q&A", page_icon="📄",
                   layout="wide", initial_sidebar_state="expanded")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MODEL_PATH = "./fine_tuned_model"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:"DM Sans",sans-serif}
.stApp{background:radial-gradient(circle at 10% 0%,rgba(124,92,255,.18),transparent 32%),radial-gradient(circle at 90% 5%,rgba(34,211,238,.12),transparent 28%),#0b1020;color:#f5f7fb}
h1,h2,h3{font-family:"Space Grotesk",sans-serif!important}
[data-testid="stSidebar"]{background:rgba(9,14,30,.96);border-right:1px solid rgba(255,255,255,.1)}
.hero{padding:2.2rem 2.4rem;border:1px solid rgba(255,255,255,.1);border-radius:24px;background:linear-gradient(135deg,rgba(124,92,255,.22),rgba(34,211,238,.08)),rgba(18,26,47,.82);box-shadow:0 20px 60px rgba(0,0,0,.25);margin-bottom:1.4rem}
.hero-badge,.model-badge{display:inline-block;padding:.35rem .75rem;border-radius:999px;background:rgba(124,92,255,.16);border:1px solid rgba(124,92,255,.35);color:#cfc5ff;font-size:.82rem;font-weight:600;margin-bottom:.8rem}
.hero h1{font-size:clamp(2rem,4vw,3.4rem);line-height:1.05;margin:0;letter-spacing:-.04em}
.hero p{color:#aab4ca;max-width:850px;font-size:1.03rem;margin-top:1rem;line-height:1.65}
.metric-card{padding:1rem 1.1rem;border:1px solid rgba(255,255,255,.1);border-radius:16px;background:rgba(18,26,47,.72)}
.metric-label{color:#aab4ca;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em}
.metric-value{color:#f5f7fb;font-size:1.18rem;font-weight:700;margin-top:.25rem}
.answer-card{padding:1.35rem 1.5rem;border:1px solid rgba(124,92,255,.3);border-radius:20px;background:linear-gradient(135deg,rgba(124,92,255,.12),rgba(18,26,47,.82));line-height:1.7;box-shadow:0 14px 45px rgba(0,0,0,.18)}
.source-card{padding:1rem 1.1rem;border:1px solid rgba(255,255,255,.1);border-radius:15px;background:rgba(23,33,59,.65)}
.small-note{color:#aab4ca;font-size:.86rem;line-height:1.55}
.stButton>button{border-radius:12px;font-weight:700}
[data-testid="stFileUploader"]{border:1px dashed rgba(124,92,255,.45);border-radius:18px;background:rgba(18,26,47,.52);padding:.5rem}
.footer{text-align:center;color:#78849d;font-size:.78rem;padding:2rem 0 1rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_fine_tuned_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=180, temperature=0.2, do_sample=True,
                    return_full_text=False, pad_token_id=tokenizer.eos_token_id)

@st.cache_resource
def load_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return [{"page": i, "text": page.extract_text() or ""}
            for i, page in enumerate(reader.pages, 1)
            if (page.extract_text() or "").strip()]

def create_chunks(pages, chunk_size=1000, overlap=200):
    chunks=[]
    step=max(1, chunk_size-overlap)
    for page in pages:
        text=page["text"]
        for start in range(0,len(text),step):
            chunk=text[start:start+chunk_size].strip()
            if chunk: chunks.append({"text":chunk,"page":page["page"]})
    return chunks

def build_faiss_index(chunks, embedding_model):
    emb=embedding_model.encode([c["text"] for c in chunks],
                               convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    index=faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

def retrieve_chunks(question,index,chunks,embedding_model,k=4):
    emb=embedding_model.encode([question],convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    scores,indices=index.search(emb,k)
    return [{"text":chunks[i]["text"],"page":chunks[i]["page"],"score":float(s)}
            for s,i in zip(scores[0],indices[0]) if i>=0]

def build_context(chunks):
    return "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)

def generate_with_huggingface(question,chunks,generator):
    prompt=f"""You are a document question-answering assistant.
Use ONLY the supplied document context. Do not invent facts.
If the answer is not contained in the context, say:
"The answer was not found in the uploaded document."

Document context:
{build_context(chunks)}

Question:
{question}

Answer:
"""
    return generator(prompt)[0]["generated_text"].strip()

def generate_with_openai(question,chunks,client):
    response=client.responses.create(
        model=OPENAI_MODEL,
        instructions=("You are a document question-answering assistant. "
                       "Answer only from the supplied PDF context. "
                       "Do not invent facts or use outside knowledge. "
                       "If the answer is not contained in the context, say: "
                       "'The answer was not found in the uploaded document.'"),
        input=f"Document context:\n\n{build_context(chunks)}\n\nQuestion:\n{question}"
    )
    return response.output_text.strip()

st.markdown("""<div class="hero">
<div class="hero-badge">GEN AI • RAG • DUAL MODEL</div>
<h1>DocuMind AI</h1>
<p>Ask natural-language questions about a PDF using semantic retrieval and
choose between your fine-tuned Hugging Face model or an OpenAI model.</p>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🤖 AI Model")
    model_choice=st.radio("Choose the answer-generation model",
                          ["Fine-Tuned Hugging Face","OpenAI"],index=0)
    st.caption("Uses your fully fine-tuned model." if model_choice=="Fine-Tuned Hugging Face"
               else f"Uses OpenAI model: {OPENAI_MODEL}")
    st.divider()
    st.markdown("## ⚙️ Architecture")
    st.markdown("PDF → Text → Chunks → Embeddings → FAISS → Retrieval → Selected LLM → Answer")
    st.divider()
    if OPENAI_API_KEY: st.success("OpenAI API key loaded from .env")
    else: st.warning("OpenAI API key not found in .env")

with st.spinner("Loading embedding model..."):
    embedding_model=load_embedding_model()

generator=None
openai_client=None
if model_choice=="Fine-Tuned Hugging Face":
    try:
        with st.spinner("Loading fine-tuned Hugging Face model..."):
            generator=load_fine_tuned_model()
    except Exception as exc:
        st.error("The fine-tuned model could not be loaded.")
        st.code(str(exc)); st.stop()
else:
    openai_client=load_openai_client()
    if openai_client is None:
        st.error("OpenAI was selected, but OPENAI_API_KEY was not found.")
        st.info("Create .env and add OPENAI_API_KEY=your_key_here")
        st.stop()

st.markdown("### 1. Upload your document")
uploaded_file=st.file_uploader("Choose a PDF document",type=["pdf"],label_visibility="collapsed")

if uploaded_file:
    signature=(uploaded_file.name,uploaded_file.size)
    if st.session_state.get("file_signature")!=signature:
        with st.spinner("Reading and indexing your PDF..."):
            pages=extract_text_from_pdf(uploaded_file)
            if not pages:
                st.error("No extractable text was found. This may be a scanned PDF requiring OCR.")
                st.stop()
            chunks=create_chunks(pages)
            index=build_faiss_index(chunks,embedding_model)
            st.session_state.update(file_signature=signature,pages=pages,chunks=chunks,index=index,
                                    answer=None,sources=[])
        st.success("Document indexed successfully.")

    c1,c2,c3=st.columns(3)
    for col,label,value in [(c1,"DOCUMENT",uploaded_file.name),
                            (c2,"PAGES",len(st.session_state.pages)),
                            (c3,"INDEXED CHUNKS",len(st.session_state.chunks))]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value">{value}</div></div>',unsafe_allow_html=True)

    st.markdown("### 2. Choose your model")
    active="Fine-Tuned Hugging Face Model" if model_choice=="Fine-Tuned Hugging Face" else f"OpenAI — {OPENAI_MODEL}"
    st.markdown(f'<div class="model-badge">Active model: {active}</div>',unsafe_allow_html=True)

    st.markdown("### 3. Ask your question")
    question=st.text_input("Question",placeholder="e.g., What are the main findings?",
                           label_visibility="collapsed")
    if st.button("✨ Generate Answer",type="primary",use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner(f"Retrieving context and generating with {active}..."):
                sources=retrieve_chunks(question,st.session_state.index,st.session_state.chunks,
                                         embedding_model,k=4)
                answer=(generate_with_huggingface(question,sources,generator)
                        if model_choice=="Fine-Tuned Hugging Face"
                        else generate_with_openai(question,sources,openai_client))
                st.session_state.answer=answer
                st.session_state.sources=sources
                st.session_state.answer_model=active

    if st.session_state.get("answer"):
        st.markdown("### Answer")
        st.markdown(f'<div class="model-badge">Generated with: {st.session_state.answer_model}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{st.session_state.answer}</div>',
                    unsafe_allow_html=True)
        st.markdown("### Retrieved Sources")
        for n,source in enumerate(st.session_state.sources,1):
            with st.expander(f"Source {n} • Page {source['page']} • Similarity {source['score']:.3f}"):
                st.markdown(f'<div class="source-card"><b>Page {source["page"]}</b><br>'
                            f'<small>Semantic similarity: {source["score"]:.4f}</small><br><br>'
                            f'{source["text"]}</div>',unsafe_allow_html=True)
else:
    st.markdown('<div class="small-note">Upload a PDF to begin. The application will extract '
                'text, create semantic embeddings, and build a searchable FAISS index.</div>',
                unsafe_allow_html=True)

st.markdown('<div class="footer">PDF-Based Question Answering System • '
            'RAG + Fine-Tuned Hugging Face + OpenAI</div>',unsafe_allow_html=True)
