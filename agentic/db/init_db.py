import asyncio
from agentic.db.models import Base
from agentic.db.session import _engine


async def main():
    async with _engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    asyncio.run(main())
