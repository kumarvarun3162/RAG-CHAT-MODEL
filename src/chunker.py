import json
import os


def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        # try to end the chunk at a newline
        if end < len(text):
            boundary = text.rfind("\n", start, end)

            if boundary > start + chunk_size // 2:
                end = boundary

        piece = text[start:end].strip()

        if len(piece) > 60:
            chunks.append({
                "id": f"chunk_{chunk_id:05d}",
                "text": piece,
                "char_start": start,
                "char_end": end
            })

            chunk_id += 1

        start = end - overlap

    return chunks


def attach_source(chunks, source_filename):
    for chunk in chunks:
        chunk["source"] = source_filename

    return chunks


def save_chunks(chunks, output_path):
    folder = os.path.dirname(output_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("Saved", len(chunks), "chunks ->", output_path)


def preview_chunks(chunks, n=3):
    print("\nTotal chunks:", len(chunks))

    if len(chunks) > 0:
        avg = sum(len(c["text"]) for c in chunks) // len(chunks)
        print("Average chunk length:", avg, "chars\n")

    for chunk in chunks[:n]:
        print(
            f'--- {chunk["id"]} '
            f'(chars {chunk["char_start"]}–{chunk["char_end"]}) ---'
        )
        print(chunk["text"][:200])
        print()


if __name__ == "__main__":

    input_file = "data/extracted_text.txt"
    output_file = "data/chunks.json"

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, 1000, 150)

    chunks = attach_source(
        chunks,
        "38300-j30.docx"
    )

    save_chunks(chunks, output_file)

    preview_chunks(chunks)