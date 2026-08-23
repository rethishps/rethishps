# DocuMind AI — PDF Question Answering

Streamlit RAG application with a model selector for:
- Fine-Tuned Hugging Face model
- OpenAI model

Both models use the same PDF retrieval pipeline (SentenceTransformers + FAISS).

## Mac setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your OpenAI key:

```text
OPENAI_API_KEY=your_actual_key
OPENAI_MODEL=gpt-4o-mini
```

Put your fine-tuned Hugging Face model files into `fine_tuned_model/`.

Run:

```bash
streamlit run app.py
```

The app is intended to run locally for the course demonstration.
