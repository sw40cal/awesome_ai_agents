import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(tool_id, name):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input={})


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.responses.pop(0))


class FakeProvider:
    def __init__(self, results):
        self.results = results
        self.sessions = []

    def handle_tool_calls(self, *, session, response):
        self.sessions.append(session)
        return self.results


class FakeSession:
    def tools(self):
        return [{"name": "COMPOSIO_SEARCH_TOOLS"}]


def test_create_session_enables_only_perplexity():
    captured = {}

    class FakeComposio:
        def create(self, **kwargs):
            captured.update(kwargs)
            return "session"

    assert agent.create_session(FakeComposio(), "user-1") == "session"
    assert captured == {"user_id": "user-1", "toolkits": ["perplexityai"]}


def test_returns_text_when_no_tool_calls():
    client = FakeClient([[text("hello")]])
    out = agent.run_agent(
        "hi", client=client, provider=FakeProvider([]), session=FakeSession()
    )
    assert out == "hello"
    assert client.calls[0]["tools"] == [{"name": "COMPOSIO_SEARCH_TOOLS"}]


def test_feeds_tool_results_back_and_finishes():
    session = FakeSession()
    client = FakeClient([[tool_use("t1", "PERPLEXITYAI_SEARCH")], [text("done")]])
    provider = FakeProvider([{"successful": False, "data": {}, "error": "boom"}])

    out = agent.run_agent("q", client=client, provider=provider, session=session)

    assert out == "done"
    assert provider.sessions == [session]
    # messages: user prompt, assistant tool_use, user tool_result, assistant text
    result = client.calls[1]["messages"][2]["content"][0]
    assert result["tool_use_id"] == "t1"
    assert result["is_error"] is True


def test_raises_after_max_turns():
    looping = [[tool_use(f"t{i}", "X")] for i in range(agent.MAX_TURNS)]
    client = FakeClient(looping)
    provider = FakeProvider([{"successful": True, "data": {}, "error": None}])
    try:
        agent.run_agent("q", client=client, provider=provider, session=FakeSession())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
