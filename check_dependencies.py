#!/usr/bin/env python3
"""
Скрипт для проверки всех зависимостей RU Crypto Bot Pro
"""

import sys
import importlib
from typing import List, Tuple

def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Проверка импорта модуля"""
    try:
        importlib.import_module(module_name)
        return True, f"✅ {package_name or module_name}"
    except ImportError as e:
        return False, f"❌ {package_name or module_name}: {str(e)}"
    except Exception as e:
        return False, f"⚠️ {package_name or module_name}: {str(e)}"

def main():
    """Основная функция проверки"""
    print("🔍 Проверка зависимостей RU Crypto Bot Pro...")
    print("=" * 50)
    
    # Список зависимостей для проверки
    dependencies = [
        ("gradio", "Gradio"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn"),
        ("ccxt", "CCXT"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("tiktoken", "TikToken"),
        ("sentencepiece", "SentencePiece"),
        ("torch.nn", "PyTorch Neural Networks"),
    ]
    
    results = []
    failed_count = 0
    
    for module, name in dependencies:
        success, message = check_import(module, name)
        results.append((success, message))
        if not success:
            failed_count += 1
        print(message)
    
    print("=" * 50)
    
    # Дополнительная информация
    try:
        import torch
        print(f"🔧 PyTorch версия: {torch.__version__}")
        print(f"🚀 CUDA доступна: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎯 CUDA версия: {torch.version.cuda}")
            print(f"📊 Количество GPU: {torch.cuda.device_count()}")
    except:
        pass
    
    try:
        import transformers
        print(f"🤖 Transformers версия: {transformers.__version__}")
    except:
        pass
    
    try:
        import tiktoken
        print(f"🔤 TikToken версия: {tiktoken.__version__}")
    except:
        pass
    
    try:
        import sys
        print(f"🐍 Python версия: {sys.version}")
    except:
        pass
    
    print("=" * 50)
    
    if failed_count == 0:
        print("🎉 Все зависимости установлены успешно!")
        print("💡 Можно запускать: python crypto_chatbot.py")
        return 0
    else:
        print(f"⚠️ Найдено проблем: {failed_count}")
        print("📦 Установите недостающие пакеты:")
        print("   pip install -r requirements.txt")
        print("\n🔧 Для решения проблемы с tiktoken:")
        print("   pip install tiktoken sentencepiece")
        return 1

if __name__ == "__main__":
    sys.exit(main())
