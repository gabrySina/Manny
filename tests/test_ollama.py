import httpx
import pytest

from manny.llm import ollama
from manny.llm.base import LLMBackendError
from manny.llm.ollama import OllamaClient, parse_version

MSG = [{"role": "user", "content": "hi"}]
SCHEMA = {"type": "object"}
REPLY = {"message": {"content": "{}"}, "prompt_eval_count": 7, "eval_count": 3}


def make_client(handler, monkeypatch):
    monkeypatch.setattr(ollama, "RETRY_PAUSE", 0)
    client = OllamaClient("http://ollama.test")
    client._http = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return client


def server(version, bodies, chat_statuses=(200,)):
    statuses = list(chat_statuses)

    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": version})
        bodies.append(__import__("json").loads(request.content))
        status = statuses.pop(0)
        return httpx.Response(status, json=REPLY if status == 200 else {"error": "boom"})

    return handler


def test_parse_version():
    assert parse_version("0.3.14") == (0, 3, 14)
    assert parse_version("0.12.1-rc2") == (0, 12, 1)
    assert parse_version("") == ()


def test_schema_is_sent_to_servers_that_support_it(monkeypatch):
    bodies = []
    result = make_client(server("0.12.3", bodies), monkeypatch).chat("m", MSG, 4096, SCHEMA, 100)
    assert bodies[0]["format"] == SCHEMA
    assert bodies[0]["options"] == {"num_ctx": 4096, "temperature": 0.2, "num_predict": 100}
    assert (result.content, result.prompt_tokens, result.completion_tokens) == ("{}", 7, 3)


def test_old_servers_get_plain_json_mode(monkeypatch):
    bodies = []
    make_client(server("0.3.14", bodies), monkeypatch).chat("m", MSG, 4096, SCHEMA)
    assert bodies[0]["format"] == "json"


def test_no_format_without_schema(monkeypatch):
    bodies = []
    make_client(server("0.12.3", bodies), monkeypatch).chat("m", MSG, 4096)
    assert "format" not in bodies[0]


def test_a_crashed_runner_is_retried_once(monkeypatch):
    bodies = []
    client = make_client(server("0.12.3", bodies, chat_statuses=(500, 200)), monkeypatch)
    assert client.chat("m", MSG, 4096).content == "{}" and len(bodies) == 2


def test_persistent_failure_raises_backend_error(monkeypatch):
    client = make_client(server("0.12.3", [], chat_statuses=(500, 500)), monkeypatch)
    with pytest.raises(LLMBackendError, match="Ollama failed on m"):
        client.chat("m", MSG, 4096)


def test_think_flag_is_sent_only_when_set(monkeypatch):
    bodies = []
    client = make_client(server("0.40.1", bodies, chat_statuses=(200, 200)), monkeypatch)
    client.chat("m", MSG, 4096)
    client.chat("m", MSG, 4096, think=False)
    assert "think" not in bodies[0] and bodies[1]["think"] is False


def test_describe_builds_an_entry_from_server_metadata(monkeypatch):
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "new:9b", "size": 6 * 1024**3}]})
        return httpx.Response(200, json={"model_info": {"general.architecture": "x"}, "capabilities": ["thinking"]})

    client = make_client(handler, monkeypatch)
    entry = client.describe("new:9b")
    assert (entry.name, entry.size_mb, entry.think) == ("new:9b", 6144, True)
    with pytest.raises(LLMBackendError, match="not installed"):
        client.describe("missing:1b")
