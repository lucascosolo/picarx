import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import llm_gateway  # noqa: E402


class _Messages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class _Anthropic:
    def __init__(self, response=None, error=None):
        self.messages = _Messages(response, error)


class _Completions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class _OpenAI:
    def __init__(self, response=None, error=None):
        self.chat = SimpleNamespace(completions=_Completions(response, error))


class GatewayTest(unittest.TestCase):
    def setUp(self):
        self.bus = harness.FakeBus()
        self.request = {
            "request_id": "r1", "task": "unit", "complexity": "low",
            "system": "be concise", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 20, "timeout": 2, "privacy": "user_request",
            "idempotent": True,
        }

    def test_normalizes_anthropic_text_and_tool_blocks(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello"),
                     SimpleNamespace(type="tool_use", id="t1", name="look",
                                      input={"pan": 2})],
            usage=SimpleNamespace(input_tokens=4, output_tokens=5),
            stop_reason="tool_use",
        )
        client = _Anthropic(response=response)
        gateway = llm_gateway.LLMGateway(bus=self.bus, anthropic_client=client)
        result = gateway.complete(**self.request)
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.content[1]["type"], "tool_use")
        self.assertEqual(result.content[1]["input"], {"pan": 2})
        self.assertEqual(result.usage["input_tokens"], 4)
        self.assertEqual(client.messages.calls[0]["model"],
                         "claude-haiku-4-5-20251001")
        status = self.bus.last(llm_gateway.STATUS_TOPIC)
        self.assertEqual(status["request_id"], "r1")
        self.assertNotIn("messages", status)

    def test_quota_failure_falls_back_to_openai(self):
        anthropic = _Anthropic(error=RuntimeError("429 quota exhausted"))
        openai_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="fallback reply", tool_calls=[]), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )
        openai = _OpenAI(response=openai_response)
        gateway = llm_gateway.LLMGateway(
            bus=self.bus, anthropic_client=anthropic, openai_client=openai)
        result = gateway.complete(**self.request)
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "openai")
        self.assertTrue(result.fallback)
        self.assertEqual(result.text, "fallback reply")
        self.assertEqual(openai.chat.completions.calls[0]["model"], "gpt-4o-mini")
        self.assertEqual(openai.chat.completions.calls[0]["messages"][0]["role"],
                         "system")

    def test_non_idempotent_failure_does_not_fallback(self):
        anthropic = _Anthropic(error=RuntimeError("temporary connection failure"))
        openai = _OpenAI(response=SimpleNamespace())
        request = dict(self.request, idempotent=False)
        gateway = llm_gateway.LLMGateway(
            bus=self.bus, anthropic_client=anthropic, openai_client=openai)
        result = gateway.complete(**request)
        self.assertFalse(result.ok)
        self.assertEqual(result.failure["code"], "claude_failed")
        self.assertEqual(openai.chat.completions.calls, [])

    def test_policy_failure_does_not_fallback(self):
        anthropic = _Anthropic(error=RuntimeError("content policy refusal"))
        openai = _OpenAI(response=SimpleNamespace())
        gateway = llm_gateway.LLMGateway(
            bus=self.bus, anthropic_client=anthropic, openai_client=openai)
        result = gateway.complete(**self.request)
        self.assertFalse(result.ok)
        self.assertEqual(result.failure["code"], "policy_refusal")
        self.assertEqual(openai.chat.completions.calls, [])

    def test_malformed_request_failure_does_not_fallback(self):
        anthropic = _Anthropic(error=RuntimeError("invalid request schema"))
        openai = _OpenAI(response=SimpleNamespace())
        gateway = llm_gateway.LLMGateway(
            bus=self.bus, anthropic_client=anthropic, openai_client=openai)
        result = gateway.complete(**self.request)
        self.assertFalse(result.ok)
        self.assertEqual(result.failure["code"], "provider_error")
        self.assertEqual(openai.chat.completions.calls, [])

    def test_local_only_request_is_blocked_before_provider(self):
        anthropic = _Anthropic(response=SimpleNamespace(content=[]))
        gateway = llm_gateway.LLMGateway(anthropic_client=anthropic)
        result = gateway.complete(**dict(self.request, privacy="local_only"))
        self.assertFalse(result.ok)
        self.assertEqual(result.failure["code"], "privacy_blocked")
        self.assertEqual(anthropic.messages.calls, [])

    def test_openai_tool_and_image_conversion(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="call1",
                    function=SimpleNamespace(name="remember", arguments='{"x": 1}'))]),
                finish_reason="tool_calls")],
            usage=None,
        )
        openai = _OpenAI(response=response)
        gateway = llm_gateway.LLMGateway(openai_client=openai)
        result = gateway.complete(
            request_id="r2", task="vision", complexity="high", system="sys",
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/jpeg", "data": "abc"}},
                {"type": "text", "text": "name it"}]}],
            tools=[{"name": "remember", "description": "save",
                    "input_schema": {"type": "object"}}],
            max_tokens=50, timeout=3, privacy="user_request", idempotent=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.content[0]["name"], "remember")
        call = openai.chat.completions.calls[0]
        self.assertEqual(call["tools"][0]["function"]["name"], "remember")
        self.assertEqual(call["messages"][1]["content"][0]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
