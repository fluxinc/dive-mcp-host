"""Database migration script."""

from logging import getLogger

from alembic import command
from alembic.config import Config

logger = getLogger(__name__)


def db_migration(
    uri: str,
    migrations_dir: str = "dive_mcp_host:httpd/database/migrations",
) -> Config:
    """Run database migrations.

    Args:
        uri: Database URI.
        migrations_dir: Migrations directory.

    Returns:
        Alembic config.
    """
    logger.info("running migration")
    config = Config()
    config.set_main_option("script_location", migrations_dir)
    config.set_main_option("sqlalchemy.url", uri)
    command.upgrade(config, "head")
    logger.info("finish migration")
    return config
