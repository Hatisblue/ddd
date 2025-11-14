"""Keyboard layouts for the bot."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton

from bot.database.models import AIModel


class Keyboards:
    """Keyboard builder class."""

    @staticmethod
    def main_menu() -> ReplyKeyboardMarkup:
        """Main menu keyboard."""
        keyboard = [
            [KeyboardButton("💬 Чат с ИИ"), KeyboardButton("🎨 Генерация изображений")],
            [KeyboardButton("🤖 Выбрать модель"), KeyboardButton("📊 Статистика")],
            [KeyboardButton("💳 Баланс"), KeyboardButton("🎁 Промокод")],
            [KeyboardButton("ℹ️ Помощь"), KeyboardButton("⚙️ Настройки")],
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    @staticmethod
    def admin_menu() -> ReplyKeyboardMarkup:
        """Admin menu keyboard."""
        keyboard = [
            [KeyboardButton("👥 Пользователи"), KeyboardButton("📈 Статистика")],
            [KeyboardButton("🎁 Промокоды"), KeyboardButton("📢 Рассылка")],
            [KeyboardButton("⬅️ Назад")],
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    @staticmethod
    def ai_model_selection() -> InlineKeyboardMarkup:
        """AI model selection keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("GPT-4o", callback_data=f"model_{AIModel.GPT4O.value}"),
                InlineKeyboardButton("GPT-4o mini", callback_data=f"model_{AIModel.GPT4O_MINI.value}"),
            ],
            [
                InlineKeyboardButton("GPT-4", callback_data=f"model_{AIModel.GPT4.value}"),
                InlineKeyboardButton("GPT-4 Turbo", callback_data=f"model_{AIModel.GPT4_TURBO.value}"),
            ],
            [
                InlineKeyboardButton("o1-preview", callback_data=f"model_{AIModel.O1_PREVIEW.value}"),
                InlineKeyboardButton("o1-mini", callback_data=f"model_{AIModel.O1_MINI.value}"),
            ],
            [
                InlineKeyboardButton("Gemini 1.5 Pro", callback_data=f"model_{AIModel.GEMINI_15_PRO.value}"),
                InlineKeyboardButton("Gemini 1.5 Flash", callback_data=f"model_{AIModel.GEMINI_15_FLASH.value}"),
            ],
            [
                InlineKeyboardButton("Grok", callback_data=f"model_{AIModel.GROK.value}"),
                InlineKeyboardButton("Grok Vision", callback_data=f"model_{AIModel.GROK_VISION.value}"),
            ],
            [InlineKeyboardButton("❌ Закрыть", callback_data="close")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def image_generation_menu() -> InlineKeyboardMarkup:
        """Image generation menu."""
        keyboard = [
            [
                InlineKeyboardButton("🎨 DALL-E 3", callback_data="image_dalle3"),
                InlineKeyboardButton("🖼 DALL-E 2", callback_data="image_dalle2"),
            ],
            [InlineKeyboardButton("⚙️ Настройки генерации", callback_data="image_settings")],
            [InlineKeyboardButton("❌ Закрыть", callback_data="close")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def image_size_selection(model: str = "dall-e-3") -> InlineKeyboardMarkup:
        """Image size selection keyboard."""
        if model == "dall-e-3":
            keyboard = [
                [InlineKeyboardButton("1024x1024 (квадрат)", callback_data="size_1024x1024")],
                [InlineKeyboardButton("1024x1792 (вертикаль)", callback_data="size_1024x1792")],
                [InlineKeyboardButton("1792x1024 (горизонталь)", callback_data="size_1792x1024")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="image_menu")],
            ]
        else:
            keyboard = [
                [InlineKeyboardButton("1024x1024", callback_data="size_1024x1024")],
                [InlineKeyboardButton("512x512", callback_data="size_512x512")],
                [InlineKeyboardButton("256x256", callback_data="size_256x256")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="image_menu")],
            ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def image_quality_selection() -> InlineKeyboardMarkup:
        """Image quality selection keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("Стандарт", callback_data="quality_standard"),
                InlineKeyboardButton("HD", callback_data="quality_hd"),
            ],
            [InlineKeyboardButton("⬅️ Назад", callback_data="image_menu")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def payment_menu() -> InlineKeyboardMarkup:
        """Payment options keyboard."""
        keyboard = [
            [InlineKeyboardButton("⭐️ Telegram Stars", callback_data="pay_stars")],
            [InlineKeyboardButton("💳 ЮКасса (карта)", callback_data="pay_yookassa")],
            [InlineKeyboardButton("🎁 Промокод", callback_data="promo_code")],
            [InlineKeyboardButton("❌ Закрыть", callback_data="close")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def token_packages() -> InlineKeyboardMarkup:
        """Token packages keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("50K токенов - 99₽", callback_data="pkg_50000_99"),
                InlineKeyboardButton("100K токенов - 179₽", callback_data="pkg_100000_179"),
            ],
            [
                InlineKeyboardButton("250K токенов - 399₽", callback_data="pkg_250000_399"),
                InlineKeyboardButton("500K токенов - 699₽", callback_data="pkg_500000_699"),
            ],
            [InlineKeyboardButton("1M токенов - 1199₽", callback_data="pkg_1000000_1199")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="balance")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def confirm_payment(package_id: str) -> InlineKeyboardMarkup:
        """Confirm payment keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("✅ Подтвердить", callback_data=f"confirm_{package_id}"),
                InlineKeyboardButton("❌ Отмена", callback_data="cancel_payment"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def settings_menu() -> InlineKeyboardMarkup:
        """Settings menu keyboard."""
        keyboard = [
            [InlineKeyboardButton("🌐 Язык", callback_data="setting_language")],
            [InlineKeyboardButton("🗑 Очистить историю", callback_data="setting_clear_history")],
            [InlineKeyboardButton("📤 Экспорт чата", callback_data="setting_export_chat")],
            [InlineKeyboardButton("❌ Закрыть", callback_data="close")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def admin_user_actions(user_id: int) -> InlineKeyboardMarkup:
        """Admin actions for user management."""
        keyboard = [
            [
                InlineKeyboardButton("➕ Добавить токены", callback_data=f"admin_add_tokens_{user_id}"),
                InlineKeyboardButton("🚫 Забанить", callback_data=f"admin_ban_{user_id}"),
            ],
            [
                InlineKeyboardButton("👑 Сделать премиум", callback_data=f"admin_premium_{user_id}"),
                InlineKeyboardButton("⚙️ Установить лимит", callback_data=f"admin_limit_{user_id}"),
            ],
            [InlineKeyboardButton("⬅️ Назад", callback_data="admin_users")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def statistics_period() -> InlineKeyboardMarkup:
        """Statistics period selection."""
        keyboard = [
            [
                InlineKeyboardButton("Сегодня", callback_data="stats_1"),
                InlineKeyboardButton("Неделя", callback_data="stats_7"),
            ],
            [
                InlineKeyboardButton("Месяц", callback_data="stats_30"),
                InlineKeyboardButton("Все время", callback_data="stats_all"),
            ],
            [InlineKeyboardButton("❌ Закрыть", callback_data="close")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def chat_mode_selection() -> InlineKeyboardMarkup:
        """Chat mode selection keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("💬 Обычный", callback_data="mode_normal"),
                InlineKeyboardButton("🎭 Креативный", callback_data="mode_creative"),
            ],
            [
                InlineKeyboardButton("💼 Деловой", callback_data="mode_professional"),
                InlineKeyboardButton("👨‍💻 Кодирование", callback_data="mode_coding"),
            ],
            [InlineKeyboardButton("❌ Закрыть", callback_data="close")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def referral_menu() -> InlineKeyboardMarkup:
        """Referral system menu."""
        keyboard = [
            [InlineKeyboardButton("📊 Мои рефералы", callback_data="ref_stats")],
            [InlineKeyboardButton("🔗 Получить ссылку", callback_data="ref_link")],
            [InlineKeyboardButton("❌ Закрыть", callback_data="close")],
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def cancel_button() -> InlineKeyboardMarkup:
        """Simple cancel button."""
        keyboard = [[InlineKeyboardButton("❌ Отмена", callback_data="cancel")]]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def close_button() -> InlineKeyboardMarkup:
        """Simple close button."""
        keyboard = [[InlineKeyboardButton("❌ Закрыть", callback_data="close")]]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def back_button(callback: str = "main_menu") -> InlineKeyboardMarkup:
        """Simple back button."""
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data=callback)]]
        return InlineKeyboardMarkup(keyboard)


# Shortcut for easier imports
kb = Keyboards()
