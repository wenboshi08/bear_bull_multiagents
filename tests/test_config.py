from bear_bull_debate.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.max_rounds == 2
    assert s.bear_model == "deepseek-v4-flash"
    assert s.bull_model == "deepseek-v4-flash"
    assert s.judge_model == "deepseek-v4-pro"
    assert s.summary_model == "deepseek-v4-flash"
    assert s.base_url == "https://api.deepseek.com"
    assert s.history_window == 4
    assert s.message_threshold == 12
    assert s.checkpointer_uri is None


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MAX_ROUNDS", "4")
    monkeypatch.setenv("JUDGE_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.example.com")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    s = Settings.from_env()
    assert s.max_rounds == 4
    assert s.judge_model == "deepseek-reasoner"
    assert s.base_url == "https://custom.example.com"
    assert s.checkpointer_uri == "postgresql://localhost/db"
