import openai, numpy as np, json, os

_client_cache = None
def _client():
    global _client_cache
    if _client_cache is None:
        _client_cache = openai.OpenAI()
    return _client_cache

def call_gpt(messages, model="gpt-4o-mini"):
    cli = _client()
    resp = cli.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()

def embed_texts(texts, model="text-embedding-3-small"):
    cli = _client()
    embs = cli.embeddings.create(model=model, input=texts).data
    return np.vstack([e.embedding for e in embs])

def load_personas(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["personas"]
