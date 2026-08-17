import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-mpnet-base-v2"
CHROMA_DIR = "vectordb"
COLLECTION = "3gpp_specs"

CONFIDENCE_THRESHOLD = 0.40


def load_retriever():

    print("Loading model and vector DB...")

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        COLLECTION
    )

    print(
        "Collection loaded:",
        collection.count(),
        "entries\n"
    )

    return model, collection


def retrieve(query, model, collection, top_k=4):

    query_vector = model.encode(
        [query]
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    similarities = []

    for d in distances:
        similarities.append(1 - d)

    retrieved = []

    for i in range(len(documents)):

        retrieved.append({
            "text": documents[i],
            "source": metadatas[i].get(
                "source",
                "unknown"
            ),
            "similarity": round(
                similarities[i],
                4
            )
        })

    return retrieved


def format_context(chunks):

    parts = []

    for i, chunk in enumerate(chunks):

        text = (
            f"[Source {i + 1} | "
            f"{chunk['source']} | "
            f"similarity: {chunk['similarity']}]\n"
            f"{chunk['text']}"
        )

        parts.append(text)

    return "\n\n---\n\n".join(parts)


def is_retrievable(chunks):

    if len(chunks) == 0:
        return False

    if chunks[0]["similarity"] >= CONFIDENCE_THRESHOLD:
        return True

    return False


if __name__ == "__main__":

    model, collection = load_retriever()

    test_queries = [
        "What is the role of the AMF in 5G architecture?",
        "How does handover work in NR?",
        "What is the capital of France?"
    ]

    for query in test_queries:

        print("Query:", query)

        results = retrieve(
            query,
            model,
            collection
        )

        if not is_retrievable(results):

            print(
                "Not found in 3GPP knowledge base.\n"
            )

            continue

        print(
            "Top similarity:",
            results[0]["similarity"]
        )

        print(
            "Retrieved text:",
            results[0]["text"][:250]
        )

        print()