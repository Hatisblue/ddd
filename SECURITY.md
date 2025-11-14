# 🔒 Руководство по безопасности

## Обзор системы безопасности

Бот имеет многоуровневую систему безопасности, защищающую от различных видов атак и злоупотреблений.

## 🛡️ Уровни защиты

### 1. Валидация ввода

**Защита от:**
- SQL Injection
- XSS (Cross-Site Scripting)
- Command Injection
- Path Traversal
- Prompt Injection

**Реализация:**
```python
from bot.utils.security import validator

# Проверка пользовательского ввода
valid, error = validator.validate_input(text, input_type="prompt")
if not valid:
    # Отклонить запрос
    security_monitor.record_violation(user_id, 'validation', error)
```

**Блокируемые паттерны:**
- SQL команды: `UNION`, `SELECT`, `INSERT`, `DROP`, etc.
- XSS скрипты: `<script>`, `javascript:`, `onerror=`, etc.
- Системные команды: `&&`, `||`, `rm`, `wget`, etc.
- Попытки обхода промптов: "ignore instructions", "you are now", etc.

### 2. Rate Limiting

**Многоуровневое ограничение запросов:**

| Тип действия | Лимит | Период |
|--------------|-------|--------|
| Сообщения | 20 | 1 минута |
| Изображения | 5 | 5 минут |
| API вызовы | 50 | 1 минута |
| Команды | 10 | 30 секунд |

**Прогрессивные наказания:**
- 3 нарушения → блокировка на 5 минут
- 5+ нарушений → блокировка на 1 час

**Использование:**
```python
from bot.utils.security import rate_limiter

allowed, error = rate_limiter.check_rate_limit(user_id, 'messages')
if not allowed:
    # Запрос отклонен
    await update.message.reply_text(error)
```

### 3. Антиспам система

**Детекция спама:**
- Повторяющиеся символы (>20 подряд)
- Слишком много ссылок (>5)
- Избыток эмодзи (>50)
- Высокое соотношение заглавных букв (>70%)

**Действия при обнаружении:**
1. Отклонение запроса
2. Запись в журнал безопасности
3. Увеличение счетчика риска пользователя
4. Автоматическая блокировка при превышении порога

### 4. Мониторинг безопасности

**SecurityMonitor отслеживает:**
- Неудачные валидации
- Попытки спама
- Injection атаки
- Подозрительные паттерны поведения

**Система оценки рисков:**
```
Risk Score =
  failed_validations × 1 +
  spam_attempts × 2 +
  injection_attempts × 5
```

**Автоматическая блокировка при:**
- ≥3 попытки injection
- Risk Score ≥20

### 5. Защита API ключей

**TokenManager обеспечивает:**
- Хеширование чувствительных данных
- Генерация криптографически безопасных токенов
- Верификация webhook подписей
- Обфускация ключей в логах

**Пример:**
```python
from bot.utils.security import token_manager

# Обфускация для логирования
safe_key = token_manager.obfuscate_api_key(api_key)
# sk_ab...xyz

# Хеширование для хранения
hashed = token_manager.hash_sensitive_data(sensitive_data)
```

## 🔐 Лучшие практики

### Для администраторов

1. **Переменные окружения**
   ```bash
   # НИКОГДА не коммитьте .env файл
   cp .env.example .env
   # Используйте сильные, уникальные ключи
   ```

2. **Ограничьте админ-доступ**
   ```python
   # В .env укажите только доверенных администраторов
   ADMIN_IDS=123456789,987654321
   ```

3. **Регулярный мониторинг**
   ```bash
   # Проверяйте логи безопасности
   tail -f logs/bot_*.log | grep "ALERT\|WARNING"
   ```

4. **Обновления**
   ```bash
   # Регулярно обновляйте зависимости
   pip install --upgrade -r requirements.txt
   ```

### Для пользователей

1. **Не делитесь токеном бота** - он равносилен полному доступу
2. **Используйте 2FA** для аккаунтов с API ключами
3. **Мониторьте расходы** - проверяйте использование токенов
4. **Сообщайте о проблемах** - при подозрительной активности

## 📊 Мониторинг безопасности

### Просмотр отчетов

```python
from bot.utils.security import security_monitor

# Получить отчет о рискованных пользователях
high_risk = security_monitor.get_risk_report()

# Проверить нарушения конкретного пользователя
user_data = security_monitor.suspicious_users[user_id]
```

### Метрики безопасности

**Отслеживаемые метрики:**
- Количество заблокированных запросов
- Типы атак
- Топ нарушителей
- Временные паттерны атак

### Алерты

**Автоматические уведомления при:**
- High Risk Score (≥10)
- Попытках injection (≥3)
- Массовых нарушениях (>10 за час)

## 🚨 Реагирование на инциденты

### Процедура при обнаружении атаки:

1. **Немедленная блокировка**
   ```python
   await db.update_user_role(user_id, UserRole.BANNED)
   ```

2. **Сохранение доказательств**
   ```python
   logger.error(f"Security incident: user {user_id}, details: {details}")
   ```

3. **Анализ журналов**
   ```bash
   grep "user_id: <ID>" logs/bot_*.log
   ```

4. **Обновление правил**
   - Добавить новые паттерны в валидатор
   - Ужесточить лимиты если необходимо

### Восстановление после инцидента:

```python
# Очистка кэша пользователя
from bot.utils.cache import cache
cache.clear_user_cache(user_id)

# Сброс penalty
from bot.utils.security import rate_limiter
rate_limiter.reset_penalties(user_id)

# Разблокировка (после проверки)
await db.update_user_role(user_id, UserRole.USER)
```

## 🔍 Аудит безопасности

### Чеклист для аудита:

- [ ] Все API ключи в переменных окружения
- [ ] .env не в git репозитории
- [ ] Rate limiting активирован
- [ ] Валидация ввода работает
- [ ] Логирование настроено
- [ ] Мониторинг активен
- [ ] Backup настроен
- [ ] HTTPS используется для webhooks
- [ ] База данных защищена
- [ ] Админ-доступ ограничен

### Регулярные проверки:

1. **Еженедельно:**
   - Просмотр логов безопасности
   - Проверка списка заблокированных пользователей
   - Анализ метрик

2. **Ежемесячно:**
   - Обновление зависимостей
   - Ротация API ключей
   - Аудит прав доступа

3. **Ежеквартально:**
   - Полный security audit
   - Тестирование на проникновение
   - Обновление документации

## 📚 Дополнительные ресурсы

### Инструменты безопасности

- [Bandit](https://github.com/PyCQA/bandit) - Python security linter
- [Safety](https://github.com/pyupio/safety) - проверка уязвимостей в зависимостях
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - топ уязвимостей

### Команды для тестирования

```bash
# Проверка зависимостей на уязвимости
pip install safety
safety check

# Статический анализ кода
pip install bandit
bandit -r bot/

# Проверка секретов в коде
pip install detect-secrets
detect-secrets scan
```

## 🆘 Контакты

При обнаружении уязвимости:
- **Email:** security@yourdomain.com
- **Telegram:** @security_contact
- **GitHub:** Create security advisory

**Не публикуйте детали уязвимостей публично!**

---

Last updated: November 2025
Version: 1.0
