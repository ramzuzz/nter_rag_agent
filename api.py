from fastapi import FastAPI
from pydantic import BaseModel
from hybrid_agent import hybrid_search_agent

# Инициализируем приложение
app = FastAPI(title="NTER RAG API")


# Описываем структуру входящего запроса
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


# Создаем эндпоинт, который будет принимать POST-запросы
@app.post("/api/chat")
def chat_with_agent(request: ChatRequest):
    print(f"Получен вопрос от {request.session_id}: {request.query}")

    try:
        # Вызываем нашу готовую функцию гибридного поиска
        answer = hybrid_search_agent(request.query, request.session_id)
        # Отдаем ответ в формате JSON
        return {"response": answer}

    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "UNAVAILABLE" in error_msg:
            return {
                "response": "Извините, серверы ИИ сейчас перегружены. Пожалуйста, подождите пару секунд и попробуйте снова! ⏳"}
        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {
                "response": "Мы превысили бесплатный лимит запросов к ИИ (Google разрешает 5 действий в минуту). Пожалуйста, подождите 40 секунд и я снова буду готов отвечать! ⏱️"}
        else:
            print(f"ОШИБКА: {error_msg}")
            return {"response": f"Техническая ошибка: {error_msg}"}