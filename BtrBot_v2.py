# семантический поиск
# кэширование страниц и ссылок
# вывод голосования под каждым слайдом
# вывод ссылок под каждым слайдом
# обрезание описания ссылок до 30 знаков
# статистика по голосованию по команде _Stats999_

# https://docs.google.com/presentation/d/1zN4KBxC-61DcMk1uG9RTOaOGcKC9UGKWC3wEDRprtF8/edit?slide=id.g2e819df673a_0_0#slide=id.g2e819df673a_0_0
# БТР - добавление новых слайдов в конец презентации, порядок слайдов 1-127 не менять и слайды не удалять

# - проверить корректность сохранения голосования в модель
# - реализовать сохранения голосования в отдельный файл для сбора статистики по слайдам
# - добавить команду для выдачи популярных слайдов

import os
import re
import fitz
import json
from config_bot import TOKEN
import torch
import pickle
import random
import logging
import asyncio
import aiofiles
import hashlib
import numpy as np
from datetime import datetime
import collections
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

    BOT_TOKEN = TOKEN

    CACHE_DIR = "cache"
    PDF_FILE_PATH = "btr3006.pdf"
    LOG_FILE_PATH = "zapros.txt"
    FEEDBACK_FILE = "feedback.pkl"
    MAX_MESSAGE_SIZE = 4000
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    QA_MODEL_NAME = "timpal0l/mdeberta-v3-base-squad2"
    CLASSIFIER_MODEL = "cointegrated/rubert-tiny-toxicity"
    SIMILARITY_THRESHOLD = 0.28
    MAX_PAGES_TO_SHOW = 5 # Лимит выдачи слайдов
    MIN_RELEVANCE_SCORE = 0.52  # Порог релевантности
    CLOSE_MATCH_CUTOFF = 0.72  # Более строгий порог для предложений
    LINKS_PER_PAGE_LIMIT = 5
    DEVICE = "cpu"
    # "cuda" if torch.cuda.is_available() else "cpu"
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
    device=-1  # if Config.DEVICE == "cuda" else -1
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

# модуль вывода статистики по голосованию по команде Stats999
PKL_FILE_PATH = 'feedback.pkl' # изменить путь на реальный на сервере


def load_feedback_data():
    """Загрузка и декодирование данных из PKL файла"""
    try:
        with open(PKL_FILE_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except UnicodeDecodeError:
        with open(PKL_FILE_PATH, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

    def convert(item):
        if isinstance(item, bytes):
            return item.decode('latin1', errors='replace')
        elif isinstance(item, list):
            return [convert(i) for i in item]
        elif isinstance(item, tuple):
            return tuple(convert(i) for i in item)
        elif isinstance(item, dict):
            return {convert(k): convert(v) for k, v in item.items()}
        elif isinstance(item, collections.defaultdict):
            return {convert(k): convert(v) for k, v in item.items()}
        return item

    return convert(data)


def aggregate_feedback_stats(decoded_data):
    """Агрегация статистики по голосам"""
    aggregation = defaultdict(lambda: [0, 0])  # [true_count, false_count]

    for key, votes in decoded_data.items():
        number = key[1]
        for vote_tuple in votes:
            if vote_tuple[1]:  # True vote
                aggregation[number][0] += 1
            else:  # False vote
                aggregation[number][1] += 1

    return dict(sorted(aggregation.items(), key=lambda x: x[0]))


def format_stats_message(aggregated_data):
    """Форматирование статистики для сообщения"""
    # Основной стиль текста


    if not aggregated_data:
        return "❌ Нет данных для отображения"

    message = "📊 <b>Статистика голосов:</b>\n\n"
    message += "<code>Слайд   |  👍 Да  |  👎 Нет  | Всего</code>\n"
    message += "<code>----------------------------------</code>\n"

    for number, (true_count, false_count) in aggregated_data.items():
        total = true_count + false_count
        message += f"<code>{number:^7} | {true_count:^6} | {false_count:^7} | {total:^5}</code>\n"

    # Итоговая статистика
    total_true = sum(true for true, _ in aggregated_data.values())
    total_false = sum(false for _, false in aggregated_data.values())
    total_all = total_true + total_false

    message += "\n<b>Итого:</b>\n"
    message += f"👍 Положительных: {total_true}\n"
    message += f"👎 Отрицательных: {total_false}\n"
    message += f"📈 Всего голосов: {total_all}"

    return message

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
        AIState.text_cache = {i + 1: text for i, text in enumerate(texts)}

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

#обрезание описания ссылки
def truncate_text(
        text: str,
        max_length: int = 30, #длинна описания ссылки
        common_prefix: str = "скотчлока. Примерный комплект для использования:"
) -> str:
    """Сокращение текста ссылки и удаление повторяющихся префиксов"""
    if not text:
        return "Ссылка без текста"

    # Удаление общего префикса (с обработкой возможных пробелов)
    clean_text = text.strip()
    prefix = common_prefix.strip()

    if clean_text.lower().startswith(prefix.lower()):
        # Удаляем префикс и все следующие за ним разделители
        clean_text = re.sub(f"^{re.escape(prefix)}[\\s,.;:]*", "", clean_text, flags=re.IGNORECASE)

    # Если текст слишком короткий - возвращаем как есть
    if len(clean_text) <= max_length:
        return clean_text or "Ссылка"

    # Сокращаем до последнего целого слова
    truncated = clean_text[:max_length]
    last_space = truncated.rfind(" ")

    if last_space > 0:
        result = truncated[:last_space] + "…"
    else:
        # Если нет пробелов, обрезаем до последнего символа и добавляем многоточие
        result = clean_text[:max_length - 1] + "…" if max_length > 1 else clean_text[0] + "…"

    return result


async def cache_page(page_number: int) -> Tuple[str, List[Tuple[str, str]]]:
    """Кэширование страницы и извлечение ссылок с текстом"""
    try:
        doc = fitz.open(Config.PDF_FILE_PATH)
        page = doc.load_page(page_number - 1)

        # Сохранение изображения страницы
        img_path = os.path.join(Config.CACHE_DIR, f"page_{page_number}.png")
        if not os.path.exists(img_path):
            pix = page.get_pixmap()
            pix.save(img_path)

        links_path = os.path.join(Config.CACHE_DIR, f"links_{page_number}.json")
        links = []

        if os.path.exists(links_path):
            async with aiofiles.open(links_path, "r") as f:
                cached_links = json.loads(await f.read())
                links = [(item["text"], item["uri"]) for item in cached_links]
        else:
            # Получаем текстовые блоки
            text_blocks = page.get_text("dict")["blocks"]
            raw_links = page.get_links()

            for link in raw_links:
                if not link.get("uri"):
                    continue
                rect = fitz.Rect(link["from"])
                link_text = ""

                # Поиск текста в области ссылки
                for block in text_blocks:
                    if block["type"] == 0:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                span_rect = fitz.Rect(span["bbox"])
                                if rect.intersects(span_rect):
                                    link_text += span["text"]

                # Чистка текста
                link_text = re.sub(r"\s+", " ", link_text).strip()
                if not link_text:
                    link_text = "Ссылка"
                else:
                    link_text = truncate_text(link_text)

                links.append((link_text, link["uri"]))  # Сохраняем (текст, URI)

            # Сохранение в кэш
            async with aiofiles.open(links_path, "w") as f:
                serialized_links = [{"text": text, "uri": uri} for text, uri in links]
                await f.write(json.dumps(serialized_links, ensure_ascii=False))

        return img_path, links[:Config.LINKS_PER_PAGE_LIMIT]
    except Exception as e:
        logging.error(f"Ошибка кэширования слайда {page_number}: {e}")
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

        return answer if answer else "Ответ не найден в БТР"
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


def create_feedback_keyboard(query: str, pages: List[int]) -> InlineKeyboardMarkup:
    """Создание клавиатуры с кнопками обратной связи и хэшем запроса"""
    builder = InlineKeyboardBuilder()
    query_hash = hashlib.md5(query.encode()).hexdigest()
    for page in pages:
        builder.row(
            InlineKeyboardButton(
                text=f"👍 Слайд {page}",
                callback_data=f"fb_{query_hash}_{page}_1"
            ),
            InlineKeyboardButton(
                text=f"👎 Слайд {page}",
                callback_data=f"fb_{query_hash}_{page}_0"
            )
        )
    return builder.as_markup()


@dp.message(CommandStart())
async def start_command(message: types.Message):
    """Обработка команды /start"""
    await message.answer(
        "🔍 Рад приветствовать вас в системе доступа к Базе Технических Решений!!\n"
        "Опишите задачу — и я подберу актуальные решения из БТР.\n\n"
        "🔎 Примеры запросов: \n"
        " • Покрытие в детской\n"
        " • Уплотнительные резинки\n\n"
        
         "Если у вас есть проблема и в БТР нет решения, напишите @VADCM,\n"
        "Либо опишите проблему в нашей группе Додо Эксплуатация в ТГ и мы поможем: \n\n"
        "https://t.me/+Nu_BhZYYqoljNmZi"

    )

# убрана функция помощи
#@dp.message(Command("help"))
#async def help_command(message: types.Message):
#    """Обработка команды /help"""
#    help_text = (
#        "🛠 **Доступные команды:**\n\n"
#        "/start - Перезапуск бота\n"
#        "/help - Справка\n"
#        "/history - История запросов\n\n"
#        "🔎 **Примеры запросов:**\n"
#        "• замена столешниц \n"
#        "• стулья \n"
#        "• резинка для холодильника"
#    )
#    await message.answer(help_text, parse_mode=ParseMode.HTML)


# функция вывода статистики по голосованиям для каждого слайда
@dp.message(Command("stats999"))
async def send_feedback_stats(message: types.Message):
    """Обработчик команды /stats999"""
    try:
        # Отправляем уведомление о начале обработки
        processing_msg = await message.answer("⏳ Загружаю данные и считаю статистику...")

        # Загрузка и обработка данных
        decoded_data = load_feedback_data()
        aggregated_data = aggregate_feedback_stats(decoded_data)
        stats_message = format_stats_message(aggregated_data)

        # Удаляем сообщение о обработке
        await bot.delete_message(
            chat_id=message.chat.id,
            message_id=processing_msg.message_id
        )

        # Отправляем результат
        await message.answer(stats_message, parse_mode="HTML")

    except Exception as e:
        error_msg = f"⚠️ Произошла ошибка: {str(e)}"
        await message.answer(error_msg)
        # Для диагностики можно добавить логирование
        print(f"Error in /stats999: {e}")


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

        response = "📜 **История ваших последних 5 запросов:**\n\n"
        for entry in user_entries[-5:]:
            parts = entry.split("; ")
            response += (
                f"📅 *{parts[0]}*\n"
                f"🔎 *Запрос:* {parts[4]}\n"
                f"📑 *Слайды:* {parts[5]}\n\n"
            )

        await message.answer(response, parse_mode=ParseMode.HTML)
    except Exception as e:
        logging.error(f"Ошибка получения истории: {e}")
        await message.answer("⚠️ Не удалось загрузить историю.")



@dp.callback_query(lambda c: c.data.startswith('fb_'))
async def handle_feedback(callback: types.CallbackQuery):
    """Обработка обратной связи"""
    try:
        _, query_hash, page, feedback = callback.data.split('_')
        page_num = int(page)
        is_helpful = bool(int(feedback))
        user = callback.from_user

        # Сохраняем обратную связь
        AIState.feedback[(query_hash, page_num)].append((user.id, is_helpful))
        await save_feedback()
        await adjust_search_weights()

        await callback.answer("Спасибо за ваш отзыв! Это улучшит результаты поиска.")
    except Exception as e:
        logging.error(f"Ошибка обработки обратной связи: {e}")
        await callback.answer("⚠️ Произошла ошибка при обработке вашего отзыва.")


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
                response += "\nВозможно, вы имели в виду:\n" + "\n".join(f"• {s}" for s in suggestions)
            await message.answer(response, parse_mode=ParseMode.HTML)
            return

        context = "\n".join(AIState.text_cache[p] for p in found_pages)
        answer = await generate_answer(query, context)

        response = (
            f"🔍 <b>Результаты по запросу:</b> <code>{escape_html(query)}</code>\n"
            f"📖 <b>Слайды из БТР: </b> {', '.join(map(str, found_pages))}"
        )

        await message.answer(response, parse_mode=ParseMode.HTML)

        for page_num in found_pages[:Config.MAX_PAGES_TO_SHOW]:
            img_path, links = await cache_page(page_num)

            if img_path and os.path.exists(img_path):
                await message.answer_photo(FSInputFile(img_path))

            formatted_links = []
            if links:
                for i, (link_text, link_url) in enumerate(links, 1):
                    if link_url and isinstance(link_url, str):
                        safe_url = escape_html(link_url.strip())
                        safe_text = escape_html(link_text.strip())
                        formatted_links.append(
                            f"{i}. <a href='{safe_url}'>{safe_text}</a>"
                        )

            if formatted_links:
                links_text = "\n".join(formatted_links)
                await message.answer(
                    f"🔗 <b>Ссылки на слайде {page_num}:</b>\n{links_text}",
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True
                )

            # Отправляем индивидуальную клавиатуру для каждой страницы
            keyboard = create_feedback_keyboard(query, [page_num])
            await message.answer(
                f"📌 Информация на слайде соответствует запросу?\n\n",
                reply_markup=keyboard

            )

            # Визуальный разделитель между слайдами
            #await message.answer("⎯⎯⎯⎯")

    except Exception as e:
        logging.error(f"Ошибка обработки запроса: {e}", exc_info=True)
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Бот остановлен пользователем.")
    except Exception as e:
        logging.error(f"Критическая ошибка при запуске бота: {e}")