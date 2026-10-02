const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.static('public')); // Отсюда будем раздавать интерфейс

const server = http.createServer(app);
const io = new Server(server);

io.on('connection', (socket) => {
    console.log('Новый пользователь подключился к чату');

    socket.on('chat_message', async (msg) => {
        console.log('Запрос от клиента:', msg);
        try {
            // Пересылаем вопрос в Python (FastAPI)
            const response = await axios.post('http://127.0.0.1:8000/api/chat', {
                query: msg,
                session_id: socket.id
            });

            // Возвращаем ответ ИИ обратно в браузер
            socket.emit('chat_response', response.data.response);
        } catch (error) {
            console.error('Ошибка бэкенда:', error.message);
            socket.emit('chat_response', 'Ошибка связи с сервером ИИ.');
        }
    });
});

server.listen(3000, () => {
    console.log('Веб-сервер запущен на http://localhost:3000');
});