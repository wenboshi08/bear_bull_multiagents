from bear_bull_debate.router import make_route_after_bull, make_route_after_summarize


def make_state(messages_len, round_, max_rounds=2):
    return {
        "messages": list(range(messages_len)),
        "round": round_,
        "max_rounds": max_rounds,
    }


def test_route_after_bull_summarize(settings):
    route = make_route_after_bull(settings)
    assert route(make_state(13, 0)) == "summarize"


def test_route_after_bull_judge(settings):
    route = make_route_after_bull(settings)
    assert route(make_state(5, 2)) == "judge"


def test_route_after_bull_continue(settings):
    route = make_route_after_bull(settings)
    assert route(make_state(5, 1)) == "bear"


def test_route_after_summarize_judge(settings):
    route = make_route_after_summarize(settings)
    assert route(make_state(0, 2)) == "judge"


def test_route_after_summarize_continue(settings):
    route = make_route_after_summarize(settings)
    assert route(make_state(0, 1)) == "bear"
