import json
import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-mpnet-base-v2"
CHROMA_DIR = "vectordb"
COLLECTION = "3gpp_specs"


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    print("Loading model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded\n")
    return model


def embed_chunks(model, chunks):
    texts = []

    for c in chunks:
        texts.append(c["text"])

    print("Encoding", len(texts), "chunks...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Embedding completed")
    print("Vector size:", embeddings.shape[1])

    return embeddings


def store_in_chromadb(chunks, embeddings):

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # remove old data if it already exists
    try:
        client.delete_collection(COLLECTION)
        print("Old collection deleted")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        ids.append(c["id"])
        documents.append(c["text"])

        metadatas.append({
            "source": c["source"],
            "char_start": c["char_start"]
        })

    batch = 500

    for i in range(0, len(chunks), batch):

        end = min(i + batch, len(chunks))

        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )

        print("Stored", end, "/", len(chunks))

    print("\nTotal entries:", collection.count())

    return collection


def sanity_check(model, collection):

    question = "What is the overall architecture of 5G NR?"

    print("\nTest question:", question)

    query_vec = model.encode(
        [question]
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=2
    )

    for i in range(len(results["documents"][0])):

        doc = results["documents"][0][i]
        distance = results["distances"][0][i]

        similarity = 1 - distance

        print("\nResult:", i + 1)
        print("Similarity:", round(similarity, 3))
        print(doc[:300])


if __name__ == "__main__":

    chunks = load_chunks("data/chunks.json")

    model = load_model()

    embeddings = embed_chunks(
        model,
        chunks
    )

    collection = store_in_chromadb(
        chunks,
        embeddings
    )

    sanity_check(
        model,
        collection
    )