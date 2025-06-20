import os
import re
import fitz
import json
import logging
import asyncio
import numpy as np
import torch
import aiofiles
import pickle
import random
import hashlib
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from difflib import get_close_matches

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.types import FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import InlineKeyboardBuilder

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

from natasha import Segmenter, Doc, NewsEmbedding, NewsMorphTagger, MorphVocab

# Конфигурация приложения
class Config:
    CACHE_DIR = "cache"
    PDF_FILE_PATH = "/opt/bot/btr_260125.pdf"
    LOG_FILE_PATH = "zapros.txt"
    FEEDBACK_FILE = "feedback.pkl"
    MAX_MESSAGE_SIZE = 4000
    BOT_TOKEN = "7532994631:AAEHa4jEpi9U8JoISi9BlNfBmn8g5ETaXtQ"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    QA_MODEL_NAME = "timpal0l/mdeberta-v3-base-squad2"
    CLASSIFIER_MODEL = "cointegrated/rubert-tiny-toxicity"
    SIMILARITY_THRESHOLD = 0.25
    MAX_PAGES_TO_SHOW = 3
    LINKS_PER_PAGE_LIMIT = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FEEDBACK_WEIGHT = 0.3
    RETRAIN_INTERVAL = 86400  # 24 часа в секундах

# Инициализация директорий
os.makedirs(Config.CACHE_DIR, exist_ok=True)

# Инициализация NLP компонентов
segmenter = Segmenter()
embedding = NewsEmbedding()
morph_tagger = NewsMorphTagger(embedding)
morph_vocab = MorphVocab()

# Инициализация AI моделей
logging.info(f"Device set to use {Config.DEVICE}")
embedder = SentenceTransformer(Config.EMBEDDING_MODEL).to(Config.DEVICE)
qa_tokenizer = AutoTokenizer.from_pretrained(Config.QA_MODEL_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(Config.QA_MODEL_NAME).to(Config.DEVICE)
classifier = pipeline(
    "text-classification",
    model=Config.CLASSIFIER_MODEL,
    device=0 if Config.DEVICE == "cuda" else -1
)

# Глобальные состояния
class AIState:
    inverse_index: Dict[str, set] = defaultdict(set)
    page_embeddings: np.ndarray = None
    text_cache: Dict[int, str] = {}
    unique_words: set = set()
    feedback: Dict[Tuple[str, int], List[Tuple[int, bool]]] = defaultdict(list)
    search_weights = {'keyword': 0.4, 'semantic': 0.4, 'feedback': 0.2}

# Инициализация бота
bot = Bot(token=Config.BOT_TOKEN)
dp = Dispatcher()

def escape_html(text: str) -> str:
    """Экранирование спецсимволов HTML"""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

async def initialize_data():
    """Инициализация данных при запуске"""
    await load_cached_data()
    if AIState.page_embeddings is None:
        await precompute_embeddings()
    if not AIState.inverse_index:
        await build_inverse_index()
    await load_feedback()
    logging.info("Data initialization completed")

async def load_cached_data():
    """Загрузка кэшированных данных"""
    try:
        if os.path.exists("page_embeddings.npy"):
            AIState.page_embeddings = np.load("page_embeddings.npy")
        if os.path.exists("text_cache.pkl"):
            with open("text_cache.pkl", "rb") as f:
                AIState.text_cache = pickle.load(f)
        if os.path.exists("inverse_index.pkl"):
            with open("inverse_index.pkl", "rb") as f:
                AIState.inverse_index = pickle.load(f)
                AIState.unique_words = set(AIState.inverse_index.keys())
    except Exception as e:
        logging.error(f"Ошибка загрузки кэша: {e}")

async def load_feedback():
    """Загрузка данных обратной связи"""
    try:
        if os.path.exists(Config.FEEDBACK_FILE):
            with open(Config.FEEDBACK_FILE, "rb") as f:
                AIState.feedback = pickle.load(f)
    except Exception as e:
        logging.error(f"Ошибка загрузки обратной связи: {e}")

async def save_feedback():
    """Сохранение данных обратной связи"""
    try:
        with open(Config.FEEDBACK_FILE, "wb") as f:
            pickle.dump(AIState.feedback, f)
    except Exception as e:
        logging.error(f"Ошибка сохранения обратной связи: {e}")

async def precompute_embeddings():
    """Предварительное вычисление эмбеддингов страниц"""
    try:
        doc = fitz.open(Config.PDF_FILE_PATH)
        texts = [page.get_text().replace("\n", " ") for page in doc]
        AIState.text_cache = {i+1: text for i, text in enumerate(texts)}
        
        embeddings = await asyncio.to_thread(
            lambda: embedder.encode(texts, convert_to_tensor=True, device=Config.DEVICE)
        )
        AIState.page_embeddings = embeddings.cpu().numpy()
        np.save("page_embeddings.npy", AIState.page_embeddings)
        
        with open("text_cache.pkl", "wb") as f:
            pickle.dump(AIState.text_cache, f)
    except Exception as e:
        logging.error(f"Ошибка вычисления эмбеддингов: {e}")

async def build_inverse_index():
    """Построение обратного индекса"""
    try:
        doc = fitz.open(Config.PDF_FILE_PATH)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            lemmas = lemmatize_text(text)
            for lemma in lemmas:
                AIState.inverse_index[lemma].add(page_num + 1)
        
        AIState.unique_words = set(AIState.inverse_index.keys())
        with open("inverse_index.pkl", "wb") as f:
            pickle.dump(AIState.inverse_index, f)
    except Exception as e:
        logging.error(f"Ошибка построения индекса: {e}")

def lemmatize_text(text: str) -> List[str]:
    """Лемматизация текста"""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    return [morph_vocab.lemmatize(token.text, token.pos, token.feats) for token in doc.tokens]

async def cache_page(page_number: int) -> Tuple[str, List[str]]:
    """Кэширование страницы и извлечение ссылок"""
    try:
        doc = fitz.open(Config.PDF_FILE_PATH)
        page = doc.load_page(page_number - 1)
        
        img_path = os.path.join(Config.CACHE_DIR, f"page_{page_number}.png")
        if not os.path.exists(img_path):
            pix = page.get_pixmap()
            pix.save(img_path)
        
        links_path = os.path.join(Config.CACHE_DIR, f"links_{page_number}.json")
        links = []
        
        if os.path.exists(links_path):
            async with aiofiles.open(links_path, "r") as f:
                links = json.loads(await f.read())
        else:
            raw_links = page.get_links()
            links = [link.get("uri", "") for link in raw_links if link.get("uri")]
            async with aiofiles.open(links_path, "w") as f:
                await f.write(json.dumps(links, ensure_ascii=False))
        
        return img_path, links[:Config.LINKS_PER_PAGE_LIMIT]
    except Exception as e:
        logging.error(f"Ошибка кэширования страницы {page_number}: {e}")
        return None, []

async def semantic_search(query: str) -> List[Tuple[int, float]]:
    """Семантический поиск"""
    try:
        query_embedding = embedder.encode([query], convert_to_tensor=True, device=Config.DEVICE)
        cos_scores = cosine_similarity(
            query_embedding.cpu(), 
            AIState.page_embeddings
        )[0]
        return sorted(enumerate(cos_scores, 1), key=lambda x: x[1], reverse=True)
    except Exception as e:
        logging.error(f"Ошибка семантического поиска: {e}")
        return []

async def keyword_search(query: str) -> List[int]:
    """Поиск по ключевым словам"""
    try:
        query_lemmas = lemmatize_text(query.lower())
        if not query_lemmas:
            return []
            
        pages = set()
        for lemma in query_lemmas:
            if lemma in AIState.inverse_index:
                if not pages:
                    pages = set(AIState.inverse_index[lemma])
                else:
                    pages &= AIState.inverse_index[lemma]
                if not pages:
                    break
        return sorted(pages)
    except Exception as e:
        logging.error(f"Ошибка keyword поиска: {e}")
        return []

async def hybrid_search(query: str) -> List[int]:
    """Гибридный поиск с учетом обратной связи"""
    try:
        keyword_pages = await keyword_search(query)
        semantic_results = await semantic_search(query)
        semantic_pages = [p for p, s in semantic_results if s > Config.SIMILARITY_THRESHOLD]
        
        feedback_scores = defaultdict(float)
        for (q, p), fbs in AIState.feedback.items():
            if q.lower() == query.lower():
                positive = sum(1 for fb in fbs if fb[1])
                total = len(fbs)
                if total > 0:
                    feedback_scores[p] = positive / total
        
        combined = set(keyword_pages + semantic_pages)
        scores = {}
        
        for page in combined:
            keyword_score = 1 if page in keyword_pages else 0
            semantic_score = next((s for p, s in semantic_results if p == page), 0)
            feedback_score = feedback_scores.get(page, 0)
            
            scores[page] = (
                AIState.search_weights['keyword'] * keyword_score +
                AIState.search_weights['semantic'] * semantic_score +
                AIState.search_weights['feedback'] * feedback_score
            )
        
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:Config.MAX_PAGES_TO_SHOW]
    except Exception as e:
        logging.error(f"Ошибка гибридного поиска: {e}")
        return []

async def generate_answer(query: str, context: str) -> str:
    """Генерация ответа"""
    try:
        inputs = qa_tokenizer(
            query, 
            context, 
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(Config.DEVICE)
        
        with torch.no_grad():
            outputs = qa_model(**inputs)
            
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        answer = qa_tokenizer.decode(
            inputs["input_ids"][0][answer_start:answer_end],
            skip_special_tokens=True
        )
        
        return answer if answer else "Ответ не найден в документе"
    except Exception as e:
        logging.error(f"Ошибка генерации ответа: {e}")
        return "Не удалось сформировать ответ"

async def log_query(user: types.User, query: str, results: List[int]):
    """Логирование запросов"""
    try:
        safe_username = escape_html(user.username or "")
        safe_fullname = escape_html(user.full_name or "")
        safe_query = escape_html(query or "")
        
        log_entry = (
            f"{datetime.now():%Y-%m-%d %H:%M:%S}; "
            f"{user.id}; "
            f"{safe_username}; "
            f"{safe_fullname}; "
            f"{safe_query}; "
            f"{', '.join(map(str, results))}\n"
        )
        
        async with aiofiles.open(Config.LOG_FILE_PATH, "a") as f:
            await f.write(log_entry)
    except Exception as e:
        logging.error(f"Ошибка логирования: {e}")

#закоментил функцию create_feedback_keyboard для отключения вывода голосования за посты
""" 
def create_feedback_keyboard(pages: List[int]) -> InlineKeyboardMarkup:
#    Создание клавиатуры с кнопками обратной связи
    builder = InlineKeyboardBuilder()
    for page in pages:
        builder.row(
            InlineKeyboardButton(
                text=f"👍 Страница {page}",
                callback_data=f"fb_{page}_1"
            ),
            InlineKeyboardButton(
                text=f"👎 Страница {page}",
                callback_data=f"fb_{page}_0"
            )
        )
    return builder.as_markup()
"""

@dp.message(CommandStart())
async def start_command(message: types.Message):
    """Обработка команды /start"""
    await message.answer(
        "🔍 Добро пожаловать в интеллектуальный поиск по технической документации!\n\n"
        "Отправьте мне ваш запрос, и я найду соответствующую информацию."
    )

@dp.message(Command("help"))
async def help_command(message: types.Message):
    """Обработка команды /help"""
    help_text = (
        "🛠 **Доступные команды:**\n\n"
        "/start - Перезапуск бота\n"
        "/help - Справка\n"
        "/history - История запросов\n\n"
        "🔎 **Примеры запросов:**\n"
        "• Настройка сервера\n"
        "• Решение ошибки 404\n"
        "• Требования безопасности"
    )
    await message.answer(help_text, parse_mode=ParseMode.HTML)

@dp.message(Command("history"))
async def history_command(message: types.Message):
    """Обработка команды /history"""
    try:
        if not os.path.exists(Config.LOG_FILE_PATH):
            await message.answer("История запросов пуста.")
            return
            
        async with aiofiles.open(Config.LOG_FILE_PATH, "r") as f:
            content = await f.read()
            
        user_entries = [
            line for line in content.splitlines() 
            if str(message.from_user.id) in line
        ]
        
        if not user_entries:
            await message.answer("Ваша история запросов пуста.")
            return
            
        response = "📜 **История ваших запросов:**\n\n"
        for entry in user_entries[-5:]:
            parts = entry.split("; ")
            response += (
                f"📅 *{parts[0]}*\n"
                f"🔎 *Запрос:* {parts[4]}\n"
                f"📑 *Страницы:* {parts[5]}\n\n"
            )
            
        await message.answer(response, parse_mode=ParseMode.HTML)
    except Exception as e:
        logging.error(f"Ошибка получения истории: {e}")
        await message.answer("⚠️ Не удалось загрузить историю.")

""" отключил для отключения голосования
@dp.callback_query(lambda c: c.data.startswith('fb_'))
async def handle_feedback(callback: types.CallbackQuery):
    Обработка обратной связи
    try:
        _, page, feedback = callback.data.split('_')
        query = callback.message.text.split("Результаты по запросу:")[1].split("\n")[0].strip().strip('`')
        
        AIState.feedback[(query, int(page))].append(
            (callback.from_user.id, bool(int(feedback)))
	)
        
        await save_feedback()
        await adjust_search_weights()
        await callback.answer("Спасибо за ваш отзыв! Это поможет улучшить результаты поиска.")
    except Exception as e:
        logging.error(f"Ошибка обработки обратной связи: {e}")
        await callback.answer("⚠️ Произошла ошибка при обработке вашего отзыва.")
"""

async def adjust_search_weights():
    """Корректировка весов поиска на основе обратной связи"""
    try:
        total_feedback = sum(len(fbs) for fbs in AIState.feedback.values())
        if total_feedback == 0:
            return

        positive = sum(1 for fbs in AIState.feedback.values() for fb in fbs if fb[1])
        positive_ratio = positive / total_feedback
        
        AIState.search_weights['feedback'] = min(0.5, positive_ratio)
        AIState.search_weights['keyword'] = 0.4 - (positive_ratio * 0.2)
        AIState.search_weights['semantic'] = 0.4 + (positive_ratio * 0.2)
        
        total = sum(AIState.search_weights.values())
        for k in AIState.search_weights:
            AIState.search_weights[k] /= total
        
        logging.info(f"Обновлены веса поиска: {AIState.search_weights}")
    except Exception as e:
        logging.error(f"Ошибка регулировки весов: {e}")

def pdf_has_changed():
    """Проверка изменений в PDF-файле"""
    try:
        current_hash = hashlib.md5(open(Config.PDF_FILE_PATH, "rb").read()).hexdigest()
        saved_hash = ""
        if os.path.exists("pdf_hash.txt"):
            with open("pdf_hash.txt", "r") as f:
                saved_hash = f.read()
        
        if current_hash != saved_hash:
            with open("pdf_hash.txt", "w") as f:
                f.write(current_hash)
            return True
        return False
    except Exception as e:
        logging.error(f"Ошибка проверки изменений PDF: {e}")
        return False

async def periodic_retraining():
    """Периодическое переобучение модели"""
    while True:
        await asyncio.sleep(Config.RETRAIN_INTERVAL)
        try:
            logging.info("Начало периодического переобучения")
            
            if pdf_has_changed():
                logging.info("Обнаружены изменения в PDF")
                await precompute_embeddings()
                await build_inverse_index()
            
            await adjust_search_weights()
            logging.info("Переобучение успешно завершено")
            
        except Exception as e:
            logging.error(f"Ошибка переобучения: {e}")

@dp.message()
async def handle_query(message: types.Message):
    """Обработка текстовых запросов"""
    try:
        user = message.from_user
        query = message.text.strip()
        
        if len(query) < 3:
            await message.answer("❌ Запрос должен содержать минимум 3 символа.")
            return
            
        toxicity = await asyncio.to_thread(classifier, query)
        if toxicity[0]['label'] == 'toxic':
            await message.answer("🚫 Запрос содержит недопустимые выражения.")
            return
            
        found_pages = await hybrid_search(query)
        await log_query(user, query, found_pages)
        
        if not found_pages:
            suggestions = get_close_matches(query, AIState.unique_words, n=5, cutoff=0.4)
            response = "🔍 По вашему запросу ничего не найдено."
            if suggestions:
                response += "\n\nВозможно, вы имели в виду:\n" + "\n".join(f"• {s}" for s in suggestions)
            await message.answer(response, parse_mode=ParseMode.HTML)
            return
            
        context = "\n".join(AIState.text_cache[p] for p in found_pages)
        answer = await generate_answer(query, context)
        
        response = (
            f"🔍 <b>Результаты по запросу:</b> <code>{escape_html(query)}</code>\n\n"
            f"{escape_html(answer)}\n\n"
            f"📖 <b>Страницы документации:</b> {', '.join(map(str, found_pages))}"
        )


#        keyboard = create_feedback_keyboard(found_pages) #убрал клавиатуру обучения
#        await message.answer(response, parse_mode=ParseMode.HTML, reply_markup=keyboard) #убрал клавиатуру обучения
        await message.answer(response, parse_mode=ParseMode.HTML) #удалить строку после возвращения клавиатуры обучения

        for page_num in found_pages[:Config.MAX_PAGES_TO_SHOW]:
            img_path, links = await cache_page(page_num)
            
            if img_path and os.path.exists(img_path):
                await message.answer_photo(FSInputFile(img_path))
            
            if links:
                formatted_links = []
                for i, link in enumerate(links, 1):
                    if link and isinstance(link, str):
                        safe_url = escape_html(link.strip())
                        formatted_links.append(
                            f"{i}. <a href='{safe_url}'>Перейти по ссылке</a>"
                        )
                
                if formatted_links:
                    links_text = "\n".join(formatted_links)
                    await message.answer(
                        f"🔗 <b>Ссылки на странице {page_num}:</b>\n{links_text}",
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True
                    )
                
    except Exception as e:
        logging.error(f"Ошибка обработки запроса: {e}")
        await message.answer("⚠️ Произошла ошибка при обработке запроса.")

async def main():
    """Основная функция"""
    await initialize_data()
    asyncio.create_task(periodic_retraining())
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    asyncio.run(main())
