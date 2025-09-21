# 🚀 Деплой RU Crypto Bot Pro на Render

## 📋 Быстрый деплой

### 1. Подготовка репозитория
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Настройка на Render.com

1. **Зайдите на [render.com](https://render.com)**
2. **Подключите GitHub репозиторий**
3. **Создайте новый Web Service**

### 3. Настройки деплоя

**Build Command:**
```bash
pip install -r requirements-render.txt
```

**Start Command:**
```bash
python crypto_chatbot.py
```

**Environment Variables:**
- `RENDER` = `true`
- `PYTHON_VERSION` = `3.9.18`

### 4. Автоматический деплой (через render.yaml)

Если в корне есть `render.yaml`, Render автоматически применит настройки.

## 🔧 Альтернативные варианты

### Вариант 1: Heroku
```bash
# Создать Procfile
echo "web: python crypto_chatbot.py" > Procfile

# Деплой
git add .
git commit -m "Add Procfile"
heroku create your-crypto-bot
git push heroku main
```

### Вариант 2: Railway
```bash
# Просто подключить GitHub репозиторий
# Railway автоматически определит Python проект
```

### Вариант 3: Docker (любая платформа)
```bash
# Собрать образ
docker build -t crypto-bot .

# Запустить локально
docker run -p 7860:7860 crypto-bot

# Деплой на любую Docker платформу
```

## ⚠️ Важные моменты

### Ресурсы
- **RAM**: Минимум 1GB (рекомендуется 2GB)
- **CPU**: 1 vCPU достаточно
- **Время загрузки**: 2-5 минут (загрузка ML моделей)

### Ограничения Free планов
- **Render Free**: 750 часов в месяц, засыпает через 15 минут неактивности
- **Heroku Free**: Отменен
- **Railway Free**: $5 кредитов в месяц

### Оптимизация для продакшена
1. **Кэширование моделей**: Модели загружаются при каждом запуске
2. **Персистентное хранилище**: Для кэша графиков
3. **CDN**: Для статических файлов

## 🐛 Устранение проблем

### Ошибка памяти
```bash
# Используйте requirements-render.txt вместо requirements.txt
# Он содержит оптимизированные зависимости
```

### Таймаут при загрузке
```bash
# Увеличьте timeout в render.yaml:
healthCheckPath: /
startCommand: python crypto_chatbot.py
```

### Модель не загружается
```bash
# Проверьте логи:
# Модель cointegrated/rut5-base-multitask может быть недоступна
# Используйте fallback модель
```

## 📊 Мониторинг

После деплоя доступно по адресу:
```
https://your-service-name.onrender.com
```

Логи доступны в панели Render Dashboard.

## 🔗 Полезные ссылки

- [Render Python Guide](https://render.com/docs/deploy-flask)
- [Gradio Deployment](https://gradio.app/sharing-your-app/)
- [GitHub Repository](https://github.com/Appstoryy/crypto_chatbot)
