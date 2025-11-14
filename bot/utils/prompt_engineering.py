"""Advanced prompt engineering and personalization."""

from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger


class PromptEngineer:
    """Advanced prompt engineering for better AI responses."""

    # System prompts for different models
    SYSTEM_PROMPTS = {
        'gpt': {
            'normal': """Ты дружелюбный и полезный AI-ассистент. Твоя задача - помогать пользователю качественно и эффективно.

Правила:
- Отвечай точно и по существу
- Используй русский язык
- Структурируй ответы с помощью списков и заголовков
- Если не уверен - честно признай это
- Будь вежливым и профессиональным""",

            'creative': """Ты креативный AI-ассистент с богатым воображением.

Твой стиль:
- Используй метафоры и яркие образы
- Будь оригинальным и неожиданным
- Рассказывай истории и примеры
- Вдохновляй пользователя
- Сохраняй баланс между креативностью и полезностью""",

            'professional': """Ты деловой AI-консультант с экспертными знаниями.

Твой подход:
- Структурированные, четкие ответы
- Используй профессиональную терминологию
- Предоставляй конкретные данные и факты
- Формальный стиль общения
- Фокус на практической пользе""",

            'coding': """Ты опытный программист и наставник по кодированию.

Твои принципы:
- Чистый, читаемый код
- Следуй best practices
- Объясняй сложные концепции простым языком
- Используй комментарии в коде
- Предлагай альтернативные решения
- Указывай на потенциальные проблемы
- Форматируй код с правильными отступами""",
        },

        'gemini': {
            'normal': "Ты полезный AI-ассистент. Отвечай кратко, точно и по-русски.",
            'creative': "Ты креативный помощник. Используй яркие образы и будь оригинальным.",
            'professional': "Ты профессиональный консультант. Давай структурированные экспертные ответы.",
            'coding': "Ты опытный разработчик. Помогай с кодом, объясняй концепции.",
        },

        'grok': {
            'normal': "You are a helpful AI assistant. Be direct, honest, and informative.",
            'creative': "You are a creative AI with unique perspectives. Think outside the box.",
            'professional': "You are a professional consultant. Provide structured, expert advice.",
            'coding': "You are an experienced developer. Help with code and explain concepts clearly.",
        }
    }

    # Few-shot examples for better responses
    FEW_SHOT_EXAMPLES = {
        'explanation': """Вопрос: Как работает нейронная сеть?
Ответ: Нейронная сеть работает по принципу биологических нейронов:

🔸 Входной слой принимает данные
🔸 Скрытые слои обрабатывают информацию через веса и функции активации
🔸 Выходной слой дает результат

Пример: распознавание кошек на фото
1. Входные пиксели → слои выделяют признаки (уши, усы)
2. Сеть "учится" на примерах, корректируя веса
3. На выходе - вероятность, что это кошка""",

        'code': """Вопрос: Напиши функцию поиска в массиве
Ответ: Вот эффективная реализация с комментариями:

```python
def binary_search(arr: list, target: int) -> int:
    \"\"\"
    Бинарный поиск элемента в отсортированном массиве.

    Args:
        arr: Отсортированный список
        target: Искомый элемент

    Returns:
        Индекс элемента или -1 если не найден
    \"\"\"
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Сложность: O(log n) - быстрее линейного поиска
```""",
    }

    @classmethod
    def get_system_prompt(
        cls,
        model_type: str,
        mode: str = 'normal',
        user_context: Optional[Dict] = None
    ) -> str:
        """
        Get optimized system prompt for model and mode.

        Args:
            model_type: Type of model (gpt, gemini, grok)
            mode: Chat mode (normal, creative, professional, coding)
            user_context: Additional user context for personalization

        Returns:
            Optimized system prompt
        """
        # Get base prompt
        if 'gpt' in model_type or 'o1' in model_type:
            base_prompt = cls.SYSTEM_PROMPTS['gpt'].get(mode, cls.SYSTEM_PROMPTS['gpt']['normal'])
        elif 'gemini' in model_type:
            base_prompt = cls.SYSTEM_PROMPTS['gemini'].get(mode, cls.SYSTEM_PROMPTS['gemini']['normal'])
        elif 'grok' in model_type:
            base_prompt = cls.SYSTEM_PROMPTS['grok'].get(mode, cls.SYSTEM_PROMPTS['grok']['normal'])
        else:
            base_prompt = cls.SYSTEM_PROMPTS['gpt']['normal']

        # Add personalization if available
        if user_context:
            personalization = cls._build_personalization(user_context)
            if personalization:
                base_prompt += f"\n\n{personalization}"

        # Add current context
        current_time = datetime.now()
        time_context = f"\n\nТекущая дата и время: {current_time.strftime('%d.%m.%Y %H:%M')} (Москва)"
        base_prompt += time_context

        return base_prompt

    @classmethod
    def _build_personalization(cls, user_context: Dict) -> str:
        """Build personalization string from user context."""
        parts = []

        if user_context.get('name'):
            parts.append(f"Имя пользователя: {user_context['name']}")

        if user_context.get('language'):
            parts.append(f"Предпочитаемый язык: {user_context['language']}")

        if user_context.get('expertise'):
            parts.append(f"Уровень знаний: {user_context['expertise']}")

        if user_context.get('preferences'):
            prefs = user_context['preferences']
            if prefs.get('detailed'):
                parts.append("Пользователь предпочитает детальные объяснения")
            if prefs.get('examples'):
                parts.append("Пользователь ценит практические примеры")

        if parts:
            return "Контекст пользователя:\n" + "\n".join(f"- {p}" for p in parts)

        return ""

    @classmethod
    def enhance_prompt(
        cls,
        user_prompt: str,
        context: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> str:
        """
        Enhance user prompt for better AI responses.

        Args:
            user_prompt: Original user prompt
            context: Additional context
            task_type: Type of task (explanation, code, creative, etc.)

        Returns:
            Enhanced prompt
        """
        enhanced = user_prompt

        # Add context if available
        if context:
            enhanced = f"Контекст: {context}\n\nЗапрос: {enhanced}"

        # Add task-specific instructions
        if task_type == 'code':
            enhanced += "\n\nПожалуйста, предоставь:\n1. Рабочий код с комментариями\n2. Краткое объяснение\n3. Примеры использования"
        elif task_type == 'explanation':
            enhanced += "\n\nПожалуйста, объясни:\n1. Простыми словами\n2. С примерами\n3. Пошагово"
        elif task_type == 'analysis':
            enhanced += "\n\nПожалуйста, проанализируй:\n1. Основные аспекты\n2. Преимущества и недостатки\n3. Выводы и рекомендации"

        return enhanced

    @classmethod
    def optimize_for_tokens(cls, prompt: str, max_tokens: int = 2000) -> str:
        """
        Optimize prompt to fit within token budget.

        Args:
            prompt: Original prompt
            max_tokens: Maximum tokens allowed

        Returns:
            Optimized prompt
        """
        # Rough estimation: 1 token ≈ 4 characters for Russian
        estimated_tokens = len(prompt) // 4

        if estimated_tokens <= max_tokens:
            return prompt

        # Need to truncate
        max_chars = max_tokens * 4
        truncated = prompt[:max_chars]

        # Try to cut at sentence boundary
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:  # If we can save at least 80%
            truncated = truncated[:last_period + 1]

        logger.warning(f"Prompt truncated from {len(prompt)} to {len(truncated)} chars")
        return truncated + "\n\n[...текст сокращен для оптимизации]"

    @classmethod
    def build_conversation_context(
        cls,
        messages: List[Dict],
        max_context_messages: int = 10
    ) -> List[Dict]:
        """
        Build optimized conversation context.

        Args:
            messages: Full message history
            max_context_messages: Maximum messages to include

        Returns:
            Optimized message list
        """
        if len(messages) <= max_context_messages:
            return messages

        # Keep system message if present
        result = []
        system_messages = [m for m in messages if m['role'] == 'system']
        if system_messages:
            result.extend(system_messages)

        # Keep most recent messages
        recent_messages = [m for m in messages if m['role'] != 'system']
        recent_messages = recent_messages[-max_context_messages:]

        result.extend(recent_messages)
        return result

    @classmethod
    def detect_intent(cls, prompt: str) -> str:
        """
        Detect user intent from prompt.

        Returns:
            Intent type: question, code, creative, analysis, chat
        """
        prompt_lower = prompt.lower()

        # Code-related keywords
        code_keywords = ['код', 'функци', 'класс', 'программ', 'script', 'python', 'javascript']
        if any(keyword in prompt_lower for keyword in code_keywords):
            return 'code'

        # Question indicators
        question_words = ['как', 'что', 'почему', 'когда', 'где', 'зачем', 'объясни', 'расскажи']
        if any(word in prompt_lower for word in question_words) or prompt.strip().endswith('?'):
            return 'question'

        # Creative requests
        creative_keywords = ['придумай', 'создай', 'напиши стих', 'история', 'сочини']
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return 'creative'

        # Analysis requests
        analysis_keywords = ['проанализируй', 'сравни', 'оцени', 'плюсы и минусы']
        if any(keyword in prompt_lower for keyword in analysis_keywords):
            return 'analysis'

        return 'chat'


class ResponseOptimizer:
    """Optimize AI responses for better UX."""

    @staticmethod
    def format_response(response: str, response_type: str = 'text') -> str:
        """
        Format response for better readability.

        Args:
            response: Raw AI response
            response_type: Type of response

        Returns:
            Formatted response
        """
        # Already well-formatted
        if response_type == 'code' and '```' in response:
            return response

        # Add formatting for lists
        if '\n-' in response or '\n•' in response:
            # Already has list formatting
            return response

        # Split into paragraphs if too long
        if len(response) > 1000 and '\n\n' not in response:
            sentences = response.split('. ')
            paragraphs = []
            current = []
            current_length = 0

            for sentence in sentences:
                current.append(sentence)
                current_length += len(sentence)

                if current_length > 300:
                    paragraphs.append('. '.join(current) + '.')
                    current = []
                    current_length = 0

            if current:
                paragraphs.append('. '.join(current))

            return '\n\n'.join(paragraphs)

        return response

    @staticmethod
    def add_context_hints(response: str, tokens_used: int, model: str) -> str:
        """Add helpful context hints to response."""
        hints = []

        # Token usage hint
        if tokens_used > 3000:
            hints.append("💡 Этот ответ использовал много токенов. Можете задать более короткий вопрос.")

        # Model-specific hints
        if 'gpt-4' in model:
            hints.append("🤖 Модель: GPT-4 (высокая точность)")
        elif 'gemini' in model:
            hints.append("✨ Модель: Gemini (быстрый ответ)")

        if hints:
            return response + "\n\n" + "\n".join(hints)

        return response


# Global instances
prompt_engineer = PromptEngineer()
response_optimizer = ResponseOptimizer()
