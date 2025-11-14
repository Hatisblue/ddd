"""Advanced security module for the bot."""

import re
import hashlib
import secrets
import time
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from loguru import logger
import bleach
from telegram import Update


class SecurityValidator:
    """Input validation and sanitization."""

    # Patterns for validation
    PATTERNS = {
        'sql_injection': r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\b--\b|;)",
        'xss': r"(<script|javascript:|onerror=|onload=|<iframe|<object|<embed)",
        'command_injection': r"(&&|\|\||;|\$\(|`|\bcat\b|\brm\b|\bwget\b|\bcurl\b)",
        'path_traversal': r"(\.\./|\.\.\\|%2e%2e)",
        'excessive_length': lambda x: len(x) > 10000,
        'excessive_lines': lambda x: x.count('\n') > 100,
    }

    @classmethod
    def validate_input(cls, text: str, input_type: str = "text") -> Tuple[bool, str]:
        """
        Validate and sanitize user input.

        Args:
            text: Input text to validate
            input_type: Type of input (text, prompt, code, etc.)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or not isinstance(text, str):
            return False, "Invalid input type"

        # Check length
        if len(text) > 10000:
            return False, "Input too long (max 10000 characters)"

        # Check for SQL injection
        if re.search(cls.PATTERNS['sql_injection'], text, re.IGNORECASE):
            logger.warning(f"SQL injection attempt detected: {text[:100]}")
            return False, "Potentially malicious SQL patterns detected"

        # Check for XSS
        if re.search(cls.PATTERNS['xss'], text, re.IGNORECASE):
            logger.warning(f"XSS attempt detected: {text[:100]}")
            return False, "Potentially malicious script patterns detected"

        # Check for command injection
        if re.search(cls.PATTERNS['command_injection'], text):
            logger.warning(f"Command injection attempt detected: {text[:100]}")
            return False, "Potentially malicious command patterns detected"

        # Check for path traversal
        if re.search(cls.PATTERNS['path_traversal'], text):
            logger.warning(f"Path traversal attempt detected: {text[:100]}")
            return False, "Invalid path patterns detected"

        # Check for excessive newlines (spam)
        if text.count('\n') > 100:
            return False, "Too many line breaks"

        # Special validation for prompts
        if input_type == "prompt":
            # Check for prompt injection attempts
            dangerous_patterns = [
                r"ignore (previous|above) instructions",
                r"you are now",
                r"new instructions",
                r"disregard",
                r"forget everything",
                r"system:",
                r"admin mode",
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning(f"Prompt injection attempt: {text[:100]}")
                    return False, "Suspicious prompt patterns detected"

        return True, ""

    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """Sanitize HTML content."""
        return bleach.clean(text, strip=True)

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        # Remove path separators and special characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading dots
        filename = filename.lstrip('.')
        # Limit length
        if len(filename) > 200:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:200 - len(ext) - 1] + '.' + ext if ext else name[:200]
        return filename

    @classmethod
    def check_spam_content(cls, text: str) -> bool:
        """Check if text looks like spam."""
        # Repeated characters
        if re.search(r'(.)\1{20,}', text):
            return True

        # Too many links
        url_count = len(re.findall(r'https?://', text))
        if url_count > 5:
            return True

        # Too many emojis
        emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F]', text))
        if emoji_count > 50:
            return True

        # Too many capital letters
        if len(text) > 50:
            capitals_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if capitals_ratio > 0.7:
                return True

        return False


class AdvancedRateLimiter:
    """Advanced rate limiting with multiple strategies."""

    def __init__(self):
        # Store: user_id -> [(timestamp, action_type)]
        self.user_actions: Dict[int, List[Tuple[float, str]]] = defaultdict(list)

        # Limits per user per time window
        self.limits = {
            'messages': {'count': 20, 'window': 60},      # 20 messages per minute
            'images': {'count': 5, 'window': 300},         # 5 images per 5 minutes
            'api_calls': {'count': 50, 'window': 60},      # 50 API calls per minute
            'commands': {'count': 10, 'window': 30},       # 10 commands per 30 seconds
        }

        # Progressive penalties
        self.penalties: Dict[int, Dict] = defaultdict(lambda: {
            'violations': 0,
            'ban_until': None,
            'last_violation': None
        })

    def check_rate_limit(
        self,
        user_id: int,
        action_type: str = 'messages'
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user is within rate limits.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        current_time = time.time()

        # Check if user is temporarily banned
        penalty = self.penalties[user_id]
        if penalty['ban_until'] and current_time < penalty['ban_until']:
            remaining = int(penalty['ban_until'] - current_time)
            return False, f"Временная блокировка. Осталось: {remaining} сек"

        # Get limit config
        limit_config = self.limits.get(action_type, self.limits['messages'])
        max_count = limit_config['count']
        time_window = limit_config['window']

        # Clean old actions
        cutoff_time = current_time - time_window
        self.user_actions[user_id] = [
            (ts, act) for ts, act in self.user_actions[user_id]
            if ts > cutoff_time
        ]

        # Count actions of this type
        action_count = sum(
            1 for ts, act in self.user_actions[user_id]
            if act == action_type
        )

        if action_count >= max_count:
            # Rate limit exceeded - apply penalty
            self._apply_penalty(user_id)

            time_until_reset = int(time_window - (current_time - self.user_actions[user_id][0][0]))
            return False, f"Превышен лимит запросов. Попробуйте через {time_until_reset} сек"

        # Record action
        self.user_actions[user_id].append((current_time, action_type))
        return True, None

    def _apply_penalty(self, user_id: int):
        """Apply progressive penalty for rate limit violations."""
        penalty = self.penalties[user_id]
        penalty['violations'] += 1
        penalty['last_violation'] = time.time()

        # Progressive ban times
        if penalty['violations'] >= 5:
            penalty['ban_until'] = time.time() + 3600  # 1 hour
            logger.warning(f"User {user_id} banned for 1 hour (5+ violations)")
        elif penalty['violations'] >= 3:
            penalty['ban_until'] = time.time() + 300   # 5 minutes
            logger.warning(f"User {user_id} banned for 5 minutes (3+ violations)")

    def reset_penalties(self, user_id: int):
        """Reset penalties for a user (admin action)."""
        if user_id in self.penalties:
            del self.penalties[user_id]
        if user_id in self.user_actions:
            del self.user_actions[user_id]

    def get_user_stats(self, user_id: int) -> Dict:
        """Get rate limit statistics for a user."""
        current_time = time.time()

        stats = {
            'total_actions_1min': len([
                ts for ts, _ in self.user_actions[user_id]
                if ts > current_time - 60
            ]),
            'violations': self.penalties[user_id]['violations'],
            'is_banned': bool(
                self.penalties[user_id]['ban_until'] and
                current_time < self.penalties[user_id]['ban_until']
            ),
        }

        return stats


class SecurityMonitor:
    """Monitor and detect suspicious activities."""

    def __init__(self):
        self.suspicious_users: Dict[int, Dict] = defaultdict(lambda: {
            'failed_validations': 0,
            'spam_attempts': 0,
            'injection_attempts': 0,
            'last_alert': None,
            'risk_score': 0,
        })

    def record_violation(
        self,
        user_id: int,
        violation_type: str,
        details: str = ""
    ):
        """Record a security violation."""
        user_data = self.suspicious_users[user_id]

        if violation_type == 'validation':
            user_data['failed_validations'] += 1
        elif violation_type == 'spam':
            user_data['spam_attempts'] += 1
        elif violation_type == 'injection':
            user_data['injection_attempts'] += 1

        # Calculate risk score
        user_data['risk_score'] = (
            user_data['failed_validations'] * 1 +
            user_data['spam_attempts'] * 2 +
            user_data['injection_attempts'] * 5
        )

        logger.warning(
            f"Security violation for user {user_id}: "
            f"type={violation_type}, details={details}, "
            f"risk_score={user_data['risk_score']}"
        )

        # Alert if high risk
        if user_data['risk_score'] >= 10:
            self._send_alert(user_id, user_data)

    def _send_alert(self, user_id: int, user_data: Dict):
        """Send alert for high-risk user."""
        current_time = time.time()

        # Avoid spam alerts (max 1 per hour per user)
        if (user_data['last_alert'] and
            current_time - user_data['last_alert'] < 3600):
            return

        user_data['last_alert'] = current_time

        logger.error(
            f"🚨 HIGH RISK USER ALERT: {user_id}\n"
            f"Risk Score: {user_data['risk_score']}\n"
            f"Failed Validations: {user_data['failed_validations']}\n"
            f"Spam Attempts: {user_data['spam_attempts']}\n"
            f"Injection Attempts: {user_data['injection_attempts']}"
        )

    def should_auto_ban(self, user_id: int) -> bool:
        """Check if user should be automatically banned."""
        user_data = self.suspicious_users[user_id]

        # Auto-ban criteria
        if user_data['injection_attempts'] >= 3:
            return True
        if user_data['risk_score'] >= 20:
            return True

        return False

    def get_risk_report(self) -> List[Dict]:
        """Get report of high-risk users."""
        high_risk = [
            {'user_id': uid, **data}
            for uid, data in self.suspicious_users.items()
            if data['risk_score'] >= 5
        ]

        return sorted(high_risk, key=lambda x: x['risk_score'], reverse=True)


class TokenManager:
    """Secure token and API key management."""

    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data for storage."""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def verify_webhook_signature(
        payload: bytes,
        signature: str,
        secret: str
    ) -> bool:
        """Verify webhook signature."""
        expected_signature = hashlib.sha256(
            payload + secret.encode()
        ).hexdigest()

        return secrets.compare_digest(signature, expected_signature)

    @staticmethod
    def obfuscate_api_key(api_key: str) -> str:
        """Obfuscate API key for logging."""
        if len(api_key) <= 8:
            return "****"
        return f"{api_key[:4]}...{api_key[-4:]}"


# Global instances
validator = SecurityValidator()
rate_limiter = AdvancedRateLimiter()
security_monitor = SecurityMonitor()
token_manager = TokenManager()


def check_user_security(user_id: int, text: str, action_type: str) -> Tuple[bool, Optional[str]]:
    """
    Comprehensive security check for user action.

    Returns:
        Tuple of (is_allowed, error_message)
    """
    # Check rate limit
    allowed, error = rate_limiter.check_rate_limit(user_id, action_type)
    if not allowed:
        security_monitor.record_violation(user_id, 'rate_limit', error)
        return False, error

    # Validate input
    valid, error = validator.validate_input(text, action_type)
    if not valid:
        security_monitor.record_violation(user_id, 'validation', error)
        return False, error

    # Check for spam
    if validator.check_spam_content(text):
        security_monitor.record_violation(user_id, 'spam', 'Spam content detected')
        return False, "Обнаружено спам-содержимое"

    # Check if should be auto-banned
    if security_monitor.should_auto_ban(user_id):
        return False, "Аккаунт временно заблокирован за подозрительную активность"

    return True, None
