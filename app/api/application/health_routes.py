from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.dependencies import get_database, get_redis_store
from app.storages.database import Database
from app.storages.redis_store import RedisStore

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(
    database: Database = Depends(get_database),
    redis_store: RedisStore = Depends(get_redis_store),
) -> dict:
    db_ok = True
    try:
        async with database.session() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    redis_ok = await redis_store.ping()

    overall = "ok" if db_ok and redis_ok else "degraded"
    return {
        "status": overall,
        "checks": {
            "database": "ok" if db_ok else "fail",
            "redis": "ok" if redis_ok else "fail",
        },
    }
