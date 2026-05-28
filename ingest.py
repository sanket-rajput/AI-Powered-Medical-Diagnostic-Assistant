import os


def ingest(source_path="data/medical_data.txt", out_dir="vectorstore"):
    try:
        from langchain.text_splitter import CharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        print("Missing required packages. Install with: pip install -r requirements.txt")
        raise

    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(source_path):
        # try pdf fallback
        base, ext = os.path.splitext(source_path)
        pdf_path = base + ".pdf"
        if os.path.exists(pdf_path):
            try:
                import PyPDF2
            except ImportError:
                print("PyPDF2 not installed. Install with: pip install PyPDF2")
                raise
            reader = PyPDF2.PdfReader(pdf_path)
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n\n".join(pages)
        else:
            print(f"Source file not found: {source_path}")
            return
    else:
        with open(source_path, "r", encoding="utf-8") as f:
            text = f.read()

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(out_dir)
    print("Vector database created successfully!")


if __name__ == "__main__":
    ingest()
