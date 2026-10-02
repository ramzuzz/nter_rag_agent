from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def test_retrieval(query):
    # 1. Загружаем ту же самую локальную модель
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 2. Подключаемся к готовой базе
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    print(f"Вопрос: {query}\n")
    print("Ищем ответы в СНиПе...\n")

    # 3. Ищем 3 самых релевантных куска текста
    docs = db.similarity_search(query, k=3)

    # 4. Выводим результаты
    for i, doc in enumerate(docs, 1):
        print(f"--- Фрагмент {i} ---")
        print(doc.page_content)
        print("-" * 50 + "\n")


if __name__ == "__main__":
    # Можешь поменять этот вопрос на любой другой по теме безопасности
    test_question = "Какие требования к защитным каскам на строительной площадке?"
    test_retrieval(test_question)