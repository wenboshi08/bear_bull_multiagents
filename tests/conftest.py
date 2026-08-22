import pytest

from bear_bull_debate.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(checkpointer_uri=None)
