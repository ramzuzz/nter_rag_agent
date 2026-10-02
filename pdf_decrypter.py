from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def create_vector_db():
    print("Загрузка PDF...")
    loader = PyPDFLoader(r"C:\Users\rasik\PycharmProjects\nter_rag_agent\docs\snip.pdf")
    documents = loader.load()

    print("Разбивка текста на фрагменты...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Создано {len(chunks)} фрагментов.")
    print("Идет скачивание и загрузка модели эмбеддингов (это займет пару минут при первом запуске)...")

    # Отличная легковесная модель, понимающая русский текст
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Сохранение в ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("Успешно! Локальная векторная база готова.")


if __name__ == "__main__":
    create_vector_db()