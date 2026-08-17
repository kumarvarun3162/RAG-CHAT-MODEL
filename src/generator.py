import requests
from src.retriever import load_retriever, retrieve, format_context, is_retrievable

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL  = "mistral"

# this prompt is your primary hallucination lock
# every rule here is deliberate — do not soften them
SYSTEM_PROMPT = """You are a precise technical assistant for 3GPP telecommunications standards.

Rules you must follow without exception:
1. Answer ONLY using the context sections provided to you below.
2. If the answer cannot be found in the context, respond with exactly:
   "This topic is not covered in the provided 3GPP documentation."
3. Do not use any knowledge from your own training about telecommunications, 5G, LTE, or related topics.
4. Always mention which source the information comes from.
5. Be technical and precise. Do not add explanations beyond what the context explicitly states.
"""


def build_prompt(query, context):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT FROM 3GPP DOCUMENTATION ---\n"
        f"{context}\n"
        f"--- END OF CONTEXT ---\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite the source section in your response):"
    )


def call_ollama(prompt):
    payload = {
        "model":  LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # low = stick to the facts, don't get creative
            "top_p":       0.9,
            "num_predict": 512,  # cap response length
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not running. Open a new terminal and run: ollama serve"
    except requests.exceptions.Timeout:
        return "ERROR: Ollama timed out. The model may still be loading — try again in 30 seconds."
    except Exception as e:
        return f"ERROR: {e}"


def answer_question(query, model, collection):
    # step 1 — embed the query and search the vector DB
    retrieved = retrieve(query, model, collection, top_k=4)

    # step 2 — if nothing is relevant enough, refuse to answer
    # this stops the LLM from hallucinating on out-of-scope questions
    if not is_retrievable(retrieved):
        return {
            "answer":   "This topic is not covered in the provided 3GPP documentation.",
            "sources":  [],
            "chunks":   []
        }

    # step 3 — format the retrieved chunks into a readable context block
    context = format_context(retrieved)

    # step 4 — build the full prompt and call the LLM
    prompt     = build_prompt(query, context)
    llm_answer = call_ollama(prompt)

    # step 5 — attach source citations to the response
    sources = [
        {"source": c["source"], "similarity": c["similarity"]}
        for c in retrieved
    ]

    return {
        "answer":  llm_answer,
        "sources": sources,
        "chunks":  retrieved
    }


if __name__ == "__main__":
    model, collection = load_retriever()

    questions = [
        "What is the role of the AMF in 5G NR?",
        "How does handover work between NR nodes?",
        "What is OFDM and how is it used in NR?",
        "What is the capital of France?",        # must be blocked
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        print("-" * 55)
        result = answer_question(q, model, collection)
        print(f"Answer:\n{result['answer']}")
        if result["sources"]:
            print(f"\nSources: {result['sources']}")
        print("=" * 55)