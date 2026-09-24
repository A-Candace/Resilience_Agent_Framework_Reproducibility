from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from agentic.common.settings import get_settings

_engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def get_session():
    async with SessionLocal() as session:
        yield session
