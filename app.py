"""
Robust FastAPI backend for AI-Chat-Bot with code-level anti-echo measures.
Replaces previous langchain-based implementation with:
- SYSTEM_PROMPT + build_prompt
- deterministic generation defaults (configurable)
- robust strip_echo() post-filter with fallback to knowledge snippets
- explicit generation kwargs usage
"""

import os
import re
import threading
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ------------------- Configs -------------------
MODEL_ID = os.getenv("MODEL_ID", "microsoft/DialoGPT-small")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.2"))
TOP_P = float(os.getenv("TOP_P", "0.7"))
DOCS_PATH = os.getenv("DOCS_PATH", os.path.join(os.path.dirname(__file__), "knowledge"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

# Sovereign system prompt (user-provided)
SYSTEM_PROMPT = (
    "Act as a Senior Sovereign Developer. Target: Vitaly Guk's creative flow. "
    "Rule: No echoing. Rule: Direct logic. Rule: 1-2 sentences max. Practical advice: Mandatory."
)

# Lazy loaded resources
tokenizer = None
model = None
pipeline_obj = None

# Simple vector store holder (FAISS if available)
embedder = None
vector_store = None
vector_store_lock = threading.Lock()

# ------------------- Utilities -------------------

def ensure_model_loaded(model_id: Optional[str] = None):
    global tokenizer, model, MODEL_ID
    if model_id is None:
        model_id = MODEL_ID
    if tokenizer is None or model is None or MODEL_ID != model_id:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        model.eval()
        MODEL_ID = model_id


def build_prompt(user_msg: str, context: str) -> str:
    # Strict instruction-first prompt (Sovereign Style)
    ctx = context.strip() if context else ""
    prompt = (
        "### INSTRUCTION:\n" + SYSTEM_PROMPT.strip() + "\n\n"
        + ("### CONTEXT:\n" + ctx + "\n\n" if ctx else "")
        + "### USER MESSAGE:\n" + user_msg.strip() + "\n\n"
        + "### RESPONSE (1-2 sentences, strictly NO ECHOING):\n"
    )
    return prompt


def build_vector_store():
    """Attempt to build FAISS index using sentence-transformers; fallback to in-memory substring store."""
    global embedder, vector_store
    with vector_store_lock:
        if vector_store is not None:
            return
        texts = []
        metadatas = []
        if os.path.isdir(DOCS_PATH):
            for fname in os.listdir(DOCS_PATH):
                if fname.endswith('.txt'):
                    try:
                        with open(os.path.join(DOCS_PATH, fname), 'r', encoding='utf-8') as f:
                            txt = f.read()
                        texts.append(txt)
                        metadatas.append({'source': fname})
                    except Exception:
                        continue
        if not texts:
            texts = [""]
            metadatas = [{'source': 'none'}]
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as _np
            import faiss as _faiss
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            emb = embedder.encode(texts, convert_to_numpy=True)
            dim = emb.shape[1]
            index = _faiss.IndexFlatL2(dim)
            index.add(emb)
            vector_store = {'index': index, 'texts': texts, 'metadatas': metadatas, 'embedder': embedder}
            return
        except Exception:
            # fallback
            vector_store = {'index': None, 'texts': texts, 'metadatas': metadatas, 'embedder': None}
            return


def similarity_search(query: str, k: int = 3):
    if vector_store is None:
        build_vector_store()
    vs = vector_store
    if vs.get('index') is not None and vs.get('embedder') is not None:
        qemb = vs['embedder'].encode([query], convert_to_numpy=True)
        D, I = vs['index'].search(qemb, k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            results.append(type('R', (), {'page_content': vs['texts'][int(idx)], 'metadata': vs['metadatas'][int(idx)], 'score': float(dist)}))
        return results
    else:
        results = []
        for t, m in zip(vs['texts'], vs['metadatas']):
            if query.lower() in (t or '').lower():
                results.append(type('R', (), {'page_content': t, 'metadata': m, 'score': 0.0}))
        return results[:k]


def generate_text(prompt: str, provider: str = "local") -> str:
    """Generate text using either local transformers or Google Gemini."""
    if provider == "gemini" and GEMINI_API_KEY:
        try:
            model_gemini = genai.GenerativeModel("gemini-pro")
            response = model_gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {e}")
            # fallback to local
    
    global pipeline_obj
    ensure_model_loaded()
    is_deterministic = float(TEMPERATURE) == 0.0
    try:
        from transformers import pipeline as _pipeline
        if pipeline_obj is None:
            dev = 0 if torch.cuda.is_available() else -1
            pipeline_obj = _pipeline('text-generation', model=model, tokenizer=tokenizer, device=dev)
        if is_deterministic:
            out = pipeline_obj(prompt, max_new_tokens=MAX_TOKENS, do_sample=False, num_beams=3, repetition_penalty=float(REPETITION_PENALTY), return_full_text=False)
        else:
            out = pipeline_obj(prompt, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=float(TEMPERATURE), top_p=float(TOP_P), repetition_penalty=float(REPETITION_PENALTY), return_full_text=False)
        if isinstance(out, list) and out:
            text = out[0].get('generated_text') or out[0].get('text') or ''
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text
    except Exception:
        pass

    # fallback to direct model.generate
    try:
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            if is_deterministic:
                out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, num_beams=3, repetition_penalty=float(REPETITION_PENALTY))
            else:
                out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=float(TEMPERATURE), top_p=float(TOP_P), repetition_penalty=float(REPETITION_PENALTY))
        gen_ids = out[0][input_len:]
        if hasattr(gen_ids, 'numel') and gen_ids.numel() > 0:
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            return text.strip()
        text_all = tokenizer.decode(out[0], skip_special_tokens=True)
        if text_all.startswith(prompt):
            return text_all[len(prompt):].strip()
        return text_all.strip()
    except Exception:
        return ''


def _normalize_text(s: str) -> str:
    if not s:
        return ''
    s = s.lower()
    s = re.sub(r"[^\w\sа-яёА-ЯЁ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_echo(user_msg: str, raw_output: str, context: str, results: List) -> str:
    """Enhanced echo suppression: physically cuts out input patterns from output."""
    out = (raw_output or '').strip()
    
    # Remove markers if model generated them
    for marker in ["### RESPONSE:", "RESPONSE:", "Answer:", "Ответ:"]:
        if marker in out:
            out = out.split(marker)[-1].strip()

    # Aggressive echo removal
    user_n = _normalize_text(user_msg)
    out_n = _normalize_text(out)
    
    # If output is too similar to input (Jaccard similarity style check)
    user_words = set(user_n.split())
    out_words = set(out_n.split())
    if user_words and out_words:
        overlap = len(user_words & out_words) / float(len(user_words))
        if overlap > 0.7 or user_n in out_n:
            # Emergency fallback to knowledge or fixed guidance
            if results and len(results) > 0:
                best = results[0].page_content.strip()
                return ' '.join(re.split(r'(?<=[.!?])\s+', best)[:2]).strip()
            return "Я здесь, чтобы направлять ваш творческий процесс. Давайте попробуем сформулировать задачу иначе, чтобы я мог дать конкретный совет по сюжету или стилю."

    # Final polish: limit length
    parts = re.split(r'(?<=[.!?])\s+', out)
    return ' '.join(parts[:2]).strip()


# ------------------- API -------------------
app = FastAPI(title='AI-Chat-Bot')

class ChatReq(BaseModel):
    user_msg: str
    chat_history: List[str] = []
    provider: Optional[str] = "local" # "local" or "gemini"


@app.post('/chat')
async def chat_endpoint(payload: ChatReq, include_hint: bool = Query(False), detail_level: int = Query(0)):
    try:
        build_vector_store()
        results = similarity_search(payload.user_msg, k=3)
        context = "\n".join([r.page_content for r in results if getattr(r, 'page_content', None)])
    except Exception:
        results = []
        context = ''

    prompt = build_prompt(payload.user_msg, context)
    raw = generate_text(prompt, provider=payload.provider)
    answer = strip_echo(payload.user_msg, raw, context, results)

    if include_hint:
        hint = {'generated_tokens': len(raw.split())}
        if detail_level >= 2:
            try:
                ensure_model_loaded()
                inputs = tokenizer(payload.user_msg, return_tensors='pt')
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = torch.softmax(logits[0, -1], dim=-1)
                topk = torch.topk(probs, k=5)
                hint['top_k'] = [ {'token': tokenizer.decode([int(idx)]), 'prob': f"{float(p):.2%}"} for idx, p in zip(topk.indices.tolist(), topk.values.tolist()) ]
            except Exception:
                hint['top_k'] = None
        return {'answer': answer, 'hint': hint}
    return {'answer': answer}


@app.post('/change-model')
async def change_model(payload: dict):
    new_id = payload.get('model_id')
    if not new_id:
        raise HTTPException(400, 'model_id is required')
    try:
        global MODEL_ID, tokenizer, model, pipeline_obj
        MODEL_ID = new_id
        tokenizer = None
        model = None
        pipeline_obj = None
        ensure_model_loaded(new_id)
        return {'status': 'ok', 'model_id': new_id}
    except Exception as e:
        raise HTTPException(500, f'Failed to load model: {e}')

