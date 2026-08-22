from bear_bull_debate.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.max_rounds == 2
    assert s.judge_model == "gpt-4o"
    assert s.history_window == 4
    assert s.message_threshold == 12
    assert s.checkpointer_uri is None


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MAX_ROUNDS", "4")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-2024-08-06")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    s = Settings.from_env()
    assert s.max_rounds == 4
    assert s.judge_model == "gpt-4o-2024-08-06"
    assert s.checkpointer_uri == "postgresql://localhost/db"
