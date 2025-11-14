"""Monitoring and alerting system."""

import time
import psutil
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger
import asyncio


class PerformanceMonitor:
    """Monitor bot performance and health."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []

        # Thresholds
        self.thresholds = {
            'response_time': 5.0,      # seconds
            'error_rate': 0.05,         # 5%
            'memory_usage': 80,         # %
            'cpu_usage': 80,            # %
        }

    def record_request(
        self,
        request_type: str,
        duration: float,
        success: bool,
        tokens: int = 0
    ):
        """Record a request for monitoring."""
        timestamp = time.time()

        self.metrics['requests'].append({
            'type': request_type,
            'duration': duration,
            'success': success,
            'tokens': tokens,
            'timestamp': timestamp
        })

        # Check thresholds
        if duration > self.thresholds['response_time']:
            self._alert(
                'slow_response',
                f"Slow response: {duration:.2f}s for {request_type}"
            )

        # Keep only last hour of data
        cutoff = timestamp - 3600
        self.metrics['requests'] = [
            r for r in self.metrics['requests']
            if r['timestamp'] > cutoff
        ]

    def get_stats(self, period_minutes: int = 60) -> Dict:
        """Get statistics for the specified period."""
        cutoff = time.time() - (period_minutes * 60)
        recent_requests = [
            r for r in self.metrics['requests']
            if r['timestamp'] > cutoff
        ]

        if not recent_requests:
            return {
                'total_requests': 0,
                'success_rate': 0,
                'avg_response_time': 0,
                'total_tokens': 0,
            }

        total = len(recent_requests)
        successful = sum(1 for r in recent_requests if r['success'])
        total_duration = sum(r['duration'] for r in recent_requests)
        total_tokens = sum(r['tokens'] for r in recent_requests)

        return {
            'total_requests': total,
            'success_rate': successful / total if total > 0 else 0,
            'avg_response_time': total_duration / total if total > 0 else 0,
            'total_tokens': total_tokens,
            'error_rate': 1 - (successful / total) if total > 0 else 0,
        }

    def get_system_health(self) -> Dict:
        """Get system health metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            health = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_usage': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'timestamp': datetime.now().isoformat(),
            }

            # Check thresholds
            if cpu_percent > self.thresholds['cpu_usage']:
                self._alert('high_cpu', f"CPU usage: {cpu_percent}%")

            if memory.percent > self.thresholds['memory_usage']:
                self._alert('high_memory', f"Memory usage: {memory.percent}%")

            return health

        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {}

    def _alert(self, alert_type: str, message: str):
        """Record an alert."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }

        self.alerts.append(alert)
        logger.warning(f"🚨 ALERT: {alert_type} - {message}")

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def get_alerts(self, severity: str = 'all') -> List[Dict]:
        """Get recent alerts."""
        if severity == 'all':
            return self.alerts[-20:]  # Last 20 alerts

        return [a for a in self.alerts[-50:] if a['type'] == severity]

    def get_health_report(self) -> str:
        """Generate health report."""
        stats = self.get_stats(60)
        system = self.get_system_health()

        report = [
            "📊 Bot Health Report",
            "",
            "Performance (last hour):",
            f"  • Total requests: {stats['total_requests']}",
            f"  • Success rate: {stats['success_rate']:.1%}",
            f"  • Avg response time: {stats['avg_response_time']:.2f}s",
            f"  • Total tokens: {stats['total_tokens']:,}",
            "",
            "System Health:",
            f"  • CPU: {system.get('cpu_usage', 0):.1f}%",
            f"  • Memory: {system.get('memory_usage', 0):.1f}%",
            f"  • Disk: {system.get('disk_usage', 0):.1f}%",
        ]

        recent_alerts = self.get_alerts()
        if recent_alerts:
            report.extend([
                "",
                f"Recent Alerts: {len(recent_alerts)}",
            ])

        return "\n".join(report)


class UsageAnalytics:
    """Track and analyze usage patterns."""

    def __init__(self):
        self.user_metrics = defaultdict(lambda: {
            'requests': 0,
            'tokens': 0,
            'errors': 0,
            'models_used': defaultdict(int),
            'last_active': None,
        })

    def record_usage(
        self,
        user_id: int,
        model: str,
        tokens: int,
        success: bool
    ):
        """Record user usage."""
        metrics = self.user_metrics[user_id]
        metrics['requests'] += 1
        metrics['tokens'] += tokens
        metrics['models_used'][model] += 1
        metrics['last_active'] = datetime.now()

        if not success:
            metrics['errors'] += 1

    def get_user_analytics(self, user_id: int) -> Dict:
        """Get analytics for a specific user."""
        return dict(self.user_metrics[user_id])

    def get_top_users(self, limit: int = 10) -> List[Dict]:
        """Get top users by usage."""
        users = [
            {'user_id': uid, **metrics}
            for uid, metrics in self.user_metrics.items()
        ]

        return sorted(users, key=lambda x: x['tokens'], reverse=True)[:limit]

    def get_model_distribution(self) -> Dict[str, int]:
        """Get distribution of model usage."""
        distribution = defaultdict(int)

        for metrics in self.user_metrics.values():
            for model, count in metrics['models_used'].items():
                distribution[model] += count

        return dict(distribution)


class CostTracker:
    """Track API costs and token usage."""

    # Approximate costs per 1K tokens (in rubles)
    COSTS = {
        'gpt-4o': 0.015,
        'gpt-4o-mini': 0.003,
        'gpt-4-turbo': 0.02,
        'gpt-4': 0.06,
        'o1-preview': 0.03,
        'o1-mini': 0.006,
        'gemini-1.5-pro': 0.007,
        'gemini-1.5-flash': 0.001,
        'grok-beta': 0.01,
        'grok-vision-beta': 0.015,
        'dall-e-3': 80.0,  # per image
        'dall-e-2': 20.0,  # per image
    }

    def __init__(self):
        self.usage = defaultdict(lambda: {
            'tokens': 0,
            'cost': 0.0,
            'requests': 0
        })

    def record_cost(
        self,
        model: str,
        tokens: int = 0,
        images: int = 0
    ):
        """Record cost for API usage."""
        cost_per_1k = self.COSTS.get(model, 0.01)

        if images > 0:
            cost = cost_per_1k * images
        else:
            cost = (tokens / 1000) * cost_per_1k

        self.usage[model]['tokens'] += tokens
        self.usage[model]['cost'] += cost
        self.usage[model]['requests'] += 1

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        return sum(data['cost'] for data in self.usage.values())

    def get_cost_breakdown(self) -> Dict:
        """Get cost breakdown by model."""
        return {
            model: {
                'cost': data['cost'],
                'tokens': data['tokens'],
                'requests': data['requests']
            }
            for model, data in self.usage.items()
        }

    def get_cost_report(self) -> str:
        """Generate cost report."""
        breakdown = self.get_cost_breakdown()
        total = self.get_total_cost()

        report = [
            "💰 Cost Report",
            "",
            f"Total Cost: {total:.2f}₽",
            "",
            "By Model:",
        ]

        for model, data in sorted(
            breakdown.items(),
            key=lambda x: x[1]['cost'],
            reverse=True
        ):
            report.append(
                f"  • {model}: {data['cost']:.2f}₽ "
                f"({data['requests']} requests, {data['tokens']:,} tokens)"
            )

        return "\n".join(report)


# Global instances
performance_monitor = PerformanceMonitor()
usage_analytics = UsageAnalytics()
cost_tracker = CostTracker()


async def health_check_loop():
    """Background task for periodic health checks."""
    while True:
        try:
            # Check system health
            health = performance_monitor.get_system_health()

            # Log if there are issues
            if health.get('cpu_usage', 0) > 80:
                logger.warning(f"High CPU usage: {health['cpu_usage']}%")

            if health.get('memory_usage', 0) > 80:
                logger.warning(f"High memory usage: {health['memory_usage']}%")

            # Sleep for 5 minutes
            await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"Health check error: {e}")
            await asyncio.sleep(60)
