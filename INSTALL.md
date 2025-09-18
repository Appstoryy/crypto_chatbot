# 🚀 Установка RU Crypto Bot Pro

## Требования
- Python 3.8 или выше
- pip (менеджер пакетов Python)

## Быстрая установка

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd gradiochat
```

### 2. Создание виртуального окружения (рекомендуется)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Запуск приложения
```bash
python crypto_chatbot.py
```

## Альтернативная установка с conda

```bash
conda create -n crypto-bot python=3.9
conda activate crypto-bot
pip install -r requirements.txt
```

## Возможные проблемы и решения

### PyTorch
Если у вас проблемы с установкой PyTorch, используйте официальный сайт:
https://pytorch.org/get-started/locally/

### CUDA (для GPU ускорения)
Для использования GPU установите CUDA-версию PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Ошибки с ccxt
Если возникают проблемы с ccxt:
```bash
pip install --upgrade ccxt
```

## Проверка установки

Запустите следующий код для проверки:
```python
import gradio as gr
import torch
import ccxt
import transformers
print("✅ Все зависимости установлены успешно!")
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
```

## Запуск в облаке

### Google Colab
```python
!pip install -r requirements.txt
!python crypto_chatbot.py
```

### Hugging Face Spaces
Загрузите файлы в Hugging Face Spaces с `app.py` как основным файлом.

## Системные требования
- RAM: минимум 4GB, рекомендуется 8GB+
- Дисковое пространство: ~2GB для всех зависимостей
- Интернет: требуется для загрузки моделей и данных с бирж
