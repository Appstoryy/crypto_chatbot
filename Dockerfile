# Dockerfile для RU Crypto Bot Pro
FROM python:3.9-slim

# Рабочая директория
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы требований
COPY requirements-render.txt requirements.txt

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY crypto_chatbot.py .
COPY check_dependencies.py .

# Переменные окружения
ENV RENDER=true
ENV PORT=7860

# Открываем порт
EXPOSE $PORT

# Команда запуска
CMD ["python", "crypto_chatbot.py"]
