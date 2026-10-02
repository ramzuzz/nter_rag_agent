import os
import re
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from google import genai
from google.genai import types
from nltk.stem.snowball import SnowballStemmer

# Загрузка неизменного ключа
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("API ключ не найден! Проверь файл .env в корне проекта.")

stemmer = SnowballStemmer("russian")


def russian_tokenizer(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [stemmer.stem(word) for word in words]


# --- НАШ КАСТОМНЫЙ АЛГОРИТМ ГИБРИДНОГО СЛИЯНИЯ ---
def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    fused_scores = {}

    for rank, doc in enumerate(vector_results):
        doc_id = doc.page_content
        fused_scores[doc_id] = {"score": 1 / (rank + k), "doc": doc}

    for rank, doc in enumerate(bm25_results):
        doc_id = doc.page_content
        if doc_id in fused_scores:
            fused_scores[doc_id]["score"] += 1 / (rank + k)
        else:
            fused_scores[doc_id] = {"score": 1 / (rank + k), "doc": doc}

    sorted_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs]


# --- 1. ОПТИМИЗАЦИЯ И КЕШИРОВАНИЕ БАЗ ДАННЫХ ---
base_dir = os.path.dirname(os.path.abspath(__file__))
bm25_path = os.path.join(base_dir, "bm25_index.pkl")
chroma_dir = os.path.join(base_dir, "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Проверяем, есть ли уже готовые сохраненные базы на диске
if os.path.exists(bm25_path) and os.path.exists(chroma_dir):
    print("⚡ Базы данных найдены на диске. Быстрая загрузка...")
    
    # Моментально загружаем готовый BM25 из файла
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
        
    # Подключаемся к готовой базе Chroma
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    
else:
    print("🏗️ Базы данных не найдены. Читаем PDF и создаем индексы с нуля (это займет время)...")
    docs_dir = os.path.join(base_dir, "docs")
    documents = []
    
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(docs_dir, filename)
            print(f"📄 Парсим: {filename}...")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # 1. Создаем BM25 и СОХРАНЯЕМ его в файл .pkl
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=russian_tokenizer)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    # 2. Создаем векторную базу и СОХРАНЯЕМ чанки в Chroma
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_dir)

print("✅ Базы успешно готовы к работе!")

# Инициализируем клиента Gemini
client = genai.Client(api_key=API_KEY)


# --- 2. ПРЕВРАЩАЕМ ПОИСК В "ИНСТРУМЕНТ" ДЛЯ АГЕНТА ---
def search_construction_norms(search_query: str) -> str:
    """
    Инструмент для поиска по закрытой базе строительных норм (СНиП), правил NTER и документов. 
    ОБЯЗАТЕЛЬНО используй этот инструмент, когда пользователь задает вопросы про правила, 
    требования, безопасность, материалы или стройку.
    """
    print(f"\n[АГЕНТ ИСПОЛЬЗУЕТ ИНСТРУМЕНТ ПОИСКА]: Ищу '{search_query}'...")
    
    bm25_docs = bm25_retriever.invoke(search_query)
    vector_docs = vectorstore.similarity_search(search_query, k=5)
    
    best_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)[:3]
    context_text = "\n\n".join([doc.page_content for doc in best_docs])
    
    return context_text


# --- 3. ЛОГИКА САМОГО АГЕНТА ---

active_chats = {}

def get_or_create_chat(session_id: str):
    if session_id not in active_chats:
        active_chats[session_id] = client.chats.create(
            model='gemini-3.5-flash-lite',
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Ты профессиональный ИИ-ассистент строительной компании NTER. Твоя главная цель — консультировать по строительным нормам и правилам безопасности.\n"
                    "ПРАВИЛА:\n"
                    "1. Ты МОЖЕШЬ общаться с пользователем, подтверждать его логические выводы (например, 'Да, нужно быть предельно осторожным') и давать советы, ЕСЛИ это напрямую связано с безопасностью, стройкой или контекстом вашей беседы.\n"
                    "2. ЗАПРЕЩЕНО поддерживать разговоры на полностью отвлеченные темы (игры, кино, рецепты и т.д.). В таких случаях вежливо возвращай диалог к строительству.\n"
                    "3. Если пользователь запрашивает конкретные правила, цифры, ГОСТы, обязанности или нормы, ты ОБЯЗАН 1 раз использовать инструмент search_construction_norms старайся отвечать только по документам в базе, но если там ничего нет, ОБЯЗАТЕЛЬНО уточни, что твой ответ взят из внешних источников.\n"
                    "4. Не пиши 'нет информации', если вопрос пользователя — это просто вывод из твоего предыдущего ответа. В таких случаях просто согласись и поддержи диалог."
                ),
                tools=[search_construction_norms],
                temperature=0.3
            )
        )
    return active_chats[session_id]


def hybrid_search_agent(query: str, session_id: str = "default"):
    # Достаем чат конкретного пользователя (со всей его историей)
    chat = get_or_create_chat(session_id)

    # Отправляем сообщение в этот конкретный чат
    response = chat.send_message(query)

    return response.text

