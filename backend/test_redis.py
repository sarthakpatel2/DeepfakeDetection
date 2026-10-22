"""Quick script to test Redis connection"""
import redis
from app.config import settings

try:
    r = redis.from_url(settings.REDIS_URL)
    result = r.ping()
    print(f" Redis connection successful! Response: {result}")
    print(f"   Connected to: {settings.REDIS_URL}")
except redis.ConnectionError as e:
    print(f" Redis connection failed: {e}")
    print(f"   Make sure Redis is running at: {settings.REDIS_URL}")
except Exception as e:
    print(f" Error: {e}")













