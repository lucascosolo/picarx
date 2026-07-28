#!/usr/bin/env python3
"""One fail-soft boundary for optional language-model providers.

Callers describe the work (task, complexity, privacy class, and a bounded
request) instead of selecting an SDK or provider.  The gateway normalizes the
small common subset used by Layer B, records provider/fallback metadata without
recording prompts or secrets, and never executes returned tools.

Anthropic remains first choice.  An OpenAI request is used only when Claude is
unavailable or returns an eligible transient/quota failure and the caller has
marked the request idempotent.  Policy/refusal failures never get silently
retried through another provider.
"""
import json
import os
import time
from dataclasses import dataclass, field
import robot_config


STATUS_TOPIC = "picarx/llm/status"
ALLOWED_COMPLEXITIES = {"low", "standard", "high"}
LOCAL_PRIVACY_CLASSES = {"local_only", "never_cloud"}


@dataclass
class LLMResult:
    """Provider-neutral response returned to a Layer B caller."""

    request_id: str
    task: str
    text: str = ""
    content: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    fallback: bool = False
    failure: dict = field(default_factory=dict)
    stop_reason: str = ""
    raw: object = None

    @property
    def ok(self):
        return not bool(self.failure)


class LLMGateway:
    """Small provider adapter with injectable clients for off-robot tests."""

    def __init__(self, bus=None, anthropic_client=None, openai_client=None):
        self.bus = bus
        self._anthropic = anthropic_client
        self._openai = openai_client
        self._anthropic_loaded = anthropic_client is not None
        self._openai_loaded = openai_client is not None
        self.publish_health()

    # ---------- configuration and availability ----------

    @staticmethod
    def _model(complexity, fallback=False, task=""):
        complexity = complexity if complexity in ALLOWED_COMPLEXITIES else "standard"
        if fallback:
            defaults = {
                "low": "gpt-4o-mini",
                "standard": "gpt-4o-mini",
                "high": "gpt-4o",
            }
            keys = {
                "low": ("openai_low_model", "gpt-4o-mini", "LLM_OPENAI_LOW_MODEL"),
                "standard": ("openai_standard_model", "gpt-4o-mini", "LLM_OPENAI_STANDARD_MODEL"),
                "high": ("openai_high_model", "gpt-4o", "LLM_OPENAI_HIGH_MODEL"),
            }
        else:
            defaults = {
                "low": "claude-haiku-4-5-20251001",
                "standard": "claude-haiku-4-5-20251001",
                "high": "claude-sonnet-5",
            }
            legacy = {
                "low": robot_config.get(
                    "companion", "intent_model", defaults["low"], env="INTENT_MODEL"),
                "standard": robot_config.get(
                    "coach", "model", defaults["standard"], env="COACH_MODEL"),
                "high": robot_config.get(
                    "companion", "model", defaults["high"], env="COMPANION_MODEL"),
            }
            if task == "reflection":
                legacy["low"] = robot_config.get(
                    "reflection", "model", legacy["low"], env="REFLECTION_MODEL")
            keys = {
                "low": ("claude_low_model", legacy["low"], "LLM_CLAUDE_LOW_MODEL"),
                "standard": ("claude_standard_model", legacy["standard"], "LLM_CLAUDE_STANDARD_MODEL"),
                "high": ("claude_high_model", legacy["high"], "LLM_CLAUDE_HIGH_MODEL"),
            }
        key, default, env = keys[complexity]
        selected = robot_config.get("llm", key, None, env=env)
        # Preserve pre-gateway per-module model overrides when the orchestrator
        # has merely materialized the new llm defaults into an older config.
        # An explicit new LLM_* environment variable always wins.
        if (not fallback and os.environ.get(env) in (None, "")
                and selected == defaults[complexity]
                and default != defaults[complexity]):
            selected = default
        return str(selected if selected is not None else default)

    def available(self):
        """Whether at least one configured provider might be usable.

        SDK imports stay lazy.  A missing SDK is handled at request time and
        can still fall through to the other provider.
        """
        return bool(os.environ.get("ANTHROPIC_API_KEY") or
                    os.environ.get("OPENAI_API_KEY") or
                    self._anthropic is not None or self._openai is not None)

    def publish_health(self):
        """Publish provider presence/configuration without exposing secrets."""
        if self.bus is None:
            return
        payload = {
            "state": "configured" if self.available() else "unavailable",
            "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY")
                                          or self._anthropic is not None),
            "openai_configured": bool(os.environ.get("OPENAI_API_KEY")
                                       or self._openai is not None),
            "models": {
                "low": self._model("low", task="health"),
                "standard": self._model("standard", task="health"),
                "high": self._model("high", task="health"),
                "fallback_low": self._model("low", fallback=True, task="health"),
                "fallback_standard": self._model("standard", fallback=True, task="health"),
                "fallback_high": self._model("high", fallback=True, task="health"),
            },
            "ts": time.time(),
        }
        try:
            self.bus.publish(STATUS_TOPIC, payload)
        except Exception:
            pass

    def _load_anthropic(self):
        if self._anthropic_loaded:
            return self._anthropic
        self._anthropic_loaded = True
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        try:
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=key)
        except Exception as exc:
            self._anthropic = None
            self._anthropic_import_error = self._safe_error(exc)
        return self._anthropic

    def _load_openai(self):
        if self._openai_loaded:
            return self._openai
        self._openai_loaded = True
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        try:
            import openai
            self._openai = openai.OpenAI(api_key=key)
        except Exception as exc:
            self._openai = None
            self._openai_import_error = self._safe_error(exc)
        return self._openai

    # ---------- public request ----------

    def complete(self, *, request_id, task, complexity, system, messages,
                 max_tokens, timeout, privacy, tools=None, idempotent=False):
        request_id = str(request_id or f"llm-{int(time.time() * 1000)}")[:96]
        task = str(task or "unspecified")[:80]
        complexity = str(complexity or "standard").lower()
        if complexity not in ALLOWED_COMPLEXITIES:
            complexity = "standard"
        privacy = str(privacy or "local_only")[:64]
        tools = list(tools or [])
        messages = list(messages or [])

        if privacy in LOCAL_PRIVACY_CLASSES:
            return self._failure(request_id, task, "privacy_blocked",
                                 "request marked local_only")

        # Validate/cap the caller-controlled bounds before an SDK sees them.
        try:
            max_tokens = max(1, min(4096, int(max_tokens)))
            timeout = max(0.5, min(60.0, float(timeout)))
        except (TypeError, ValueError):
            return self._failure(request_id, task, "invalid_limits",
                                 "invalid request limits")

        first = self._load_anthropic()
        primary_error = None
        if first is not None:
            result = self._call_anthropic(
                first, request_id, task, complexity, system, messages, tools,
                max_tokens, timeout, fallback=False)
            if result.ok:
                return result
            primary_error = result.failure
            if not self._eligible_fallback(primary_error):
                return result
        else:
            primary_error = {"code": "claude_unavailable",
                             "message": getattr(self, "_anthropic_import_error", "missing key or SDK")}

        # A failed model response must not cause a second provider to execute
        # a non-idempotent request. Tool execution is owned by callers and is
        # therefore safe only when the whole model request is explicitly
        # marked idempotent.
        if not idempotent:
            return self._failure(request_id, task, "claude_failed",
                                 primary_error.get("message", "Claude failed"),
                                 provider="anthropic", detail=primary_error)

        fallback = self._load_openai()
        if fallback is None:
            return self._failure(request_id, task, "no_fallback_provider",
                                 primary_error.get("message", "no provider available"),
                                 provider="anthropic", detail=primary_error)
        result = self._call_openai(
            fallback, request_id, task, complexity, system, messages, tools,
            max_tokens, timeout, fallback=True)
        if not result.ok:
            result.failure = {
                "code": "providers_failed",
                "message": result.failure.get("message", "OpenAI failed"),
                "primary": primary_error,
                "fallback": result.failure,
            }
        return result

    # ---------- provider calls ----------

    def _call_anthropic(self, client, request_id, task, complexity, system,
                        messages, tools, max_tokens, timeout, fallback):
        model = self._model(complexity, fallback=False, task=task)
        started = time.monotonic()
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "system": str(system or ""),
                "messages": self._anthropic_messages(messages),
                "timeout": timeout,
            }
            if tools:
                kwargs["tools"] = tools
            response = client.messages.create(**kwargs)
            content = self._anthropic_content(response)
            result = LLMResult(
                request_id=request_id, task=task,
                text=self._text(content), content=content,
                usage=self._usage(getattr(response, "usage", None)),
                provider="anthropic", model=model,
                latency_ms=self._latency(started), fallback=fallback,
                stop_reason=str(getattr(response, "stop_reason", "") or ""),
                raw=response)
            self._publish(result)
            return result
        except Exception as exc:
            result = self._failure(request_id, task, self._error_code(exc),
                                   self._safe_error(exc), provider="anthropic",
                                   model=model, latency_ms=self._latency(started),
                                   fallback=fallback)
            self._publish(result)
            return result

    def _call_openai(self, client, request_id, task, complexity, system,
                     messages, tools, max_tokens, timeout, fallback):
        model = self._model(complexity, fallback=True, task=task)
        started = time.monotonic()
        try:
            kwargs = {
                "model": model,
                "messages": self._openai_messages(system, messages),
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if tools:
                kwargs["tools"] = self._openai_tools(tools)
            response = client.chat.completions.create(**kwargs)
            content = self._openai_content(response)
            result = LLMResult(
                request_id=request_id, task=task,
                text=self._text(content), content=content,
                usage=self._usage(getattr(response, "usage", None)),
                provider="openai", model=model,
                latency_ms=self._latency(started), fallback=fallback,
                stop_reason=self._openai_stop_reason(response), raw=response)
            self._publish(result)
            return result
        except Exception as exc:
            result = self._failure(request_id, task, self._error_code(exc),
                                   self._safe_error(exc), provider="openai",
                                   model=model, latency_ms=self._latency(started),
                                   fallback=fallback)
            self._publish(result)
            return result

    # ---------- normalized message/response shapes ----------

    @staticmethod
    def _value(value, key, default=None):
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    @classmethod
    def _blocks(cls, content):
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, (tuple, list)):
            out = []
            for block in content:
                if isinstance(block, dict):
                    out.append(dict(block))
                    continue
                kind = cls._value(block, "type")
                if kind == "text":
                    out.append({"type": "text", "text": cls._value(block, "text", "")})
                elif kind in ("tool_use", "function"):
                    name = cls._value(block, "name")
                    arguments = cls._value(block, "input", cls._value(block, "arguments", {}))
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (TypeError, ValueError):
                            arguments = {}
                    out.append({"type": "tool_use",
                                "id": cls._value(block, "id", ""),
                                "name": name or "",
                                "input": arguments or {}})
            return out
        return []

    @classmethod
    def _anthropic_content(cls, response):
        return cls._blocks(cls._value(response, "content", []))

    @classmethod
    def _openai_content(cls, response):
        choices = cls._value(response, "choices", []) or []
        if not choices:
            return []
        message = cls._value(choices[0], "message", None)
        blocks = cls._blocks(cls._value(message, "content", ""))
        for call in cls._value(message, "tool_calls", []) or []:
            function = cls._value(call, "function", None)
            args = cls._value(function, "arguments", "{}")
            try:
                args = json.loads(args) if isinstance(args, str) else (args or {})
            except (TypeError, ValueError):
                args = {}
            blocks.append({"type": "tool_use",
                           "id": cls._value(call, "id", ""),
                           "name": cls._value(function, "name", ""),
                           "input": args})
        return blocks

    @classmethod
    def _openai_stop_reason(cls, response):
        choices = cls._value(response, "choices", []) or []
        return str(cls._value(choices[0], "finish_reason", "") or "") if choices else ""

    @classmethod
    def _text(cls, content):
        return "".join(str(b.get("text", "")) for b in content
                        if b.get("type") == "text").strip()

    @classmethod
    def _anthropic_messages(cls, messages):
        out = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
            content = message.get("content", "")
            blocks = cls._blocks(content)
            # Preserve Anthropic's native shape; dict blocks are already the
            # neutral shape accepted by its Messages API.
            out.append({"role": role, "content": blocks if isinstance(content, (list, tuple)) else content})
        return out

    @classmethod
    def _openai_messages(cls, system, messages):
        out = [{"role": "system", "content": str(system or "")}]
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "assistant" and isinstance(content, (list, tuple)):
                blocks = cls._blocks(content)
                text = cls._text(blocks)
                calls = [
                    {"id": b.get("id", ""), "type": "function",
                     "function": {"name": b.get("name", ""),
                                  "arguments": json.dumps(b.get("input") or {})}}
                    for b in blocks if b.get("type") == "tool_use"
                ]
                item = {"role": "assistant", "content": text or None}
                if calls:
                    item["tool_calls"] = calls
                out.append(item)
                continue
            if role == "user" and isinstance(content, (list, tuple)):
                ordinary = []
                for block in cls._blocks(content):
                    kind = block.get("type")
                    if kind == "tool_result":
                        out.append({"role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": str(block.get("content", ""))})
                    elif kind == "image":
                        source = block.get("source") or {}
                        if source.get("type") == "base64":
                            ordinary.append({"type": "image_url",
                                             "image_url": {"url": "data:%s;base64,%s" %
                                                            (source.get("media_type", "image/jpeg"),
                                                             source.get("data", ""))}})
                    elif kind == "text":
                        ordinary.append({"type": "text", "text": block.get("text", "")})
                    else:
                        ordinary.append(block)
                out.append({"role": "user", "content": ordinary})
                continue
            out.append({"role": role, "content": content})
        return out

    @staticmethod
    def _openai_tools(tools):
        out = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            out.append({"type": "function", "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            }})
        return out

    # ---------- bounded diagnostics ----------

    @staticmethod
    def _usage(usage):
        if usage is None:
            return {}
        out = {}
        for source, target in (("input_tokens", "input_tokens"),
                               ("output_tokens", "output_tokens"),
                               ("prompt_tokens", "input_tokens"),
                               ("completion_tokens", "output_tokens"),
                               ("total_tokens", "total_tokens")):
            value = usage.get(source) if isinstance(usage, dict) else getattr(usage, source, None)
            if value is not None and target not in out:
                out[target] = value
        return out

    @staticmethod
    def _latency(started):
        return round((time.monotonic() - started) * 1000.0, 1)

    @staticmethod
    def _safe_error(exc):
        text = str(exc) or exc.__class__.__name__
        for secret in (os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("OPENAI_API_KEY")):
            if secret:
                text = text.replace(secret, "[redacted]")
        return text[:300]

    @classmethod
    def _error_code(cls, exc):
        text = cls._safe_error(exc).lower()
        kind = exc.__class__.__name__.lower()
        if ("rate" in kind and "limit" in kind) or any(
                x in text for x in ("quota", "rate limit", "rate_limit", "429", "too many")):
            return "quota_or_rate_limit"
        if "timeout" in kind or any(x in text for x in ("timeout", "timed out", "deadline")):
            return "timeout"
        if any(x in text for x in ("policy", "safety", "refusal", "content filter")):
            return "policy_refusal"
        if any(x in text for x in ("connection", "temporar", "503", "502", "500", "unavailable")):
            return "transient_provider_error"
        return "provider_error"

    @staticmethod
    def _eligible_fallback(failure):
        return failure.get("code") in {
            "claude_unavailable", "quota_or_rate_limit", "timeout",
            "transient_provider_error",
        }

    def _failure(self, request_id, task, code, message, provider="", model="",
                 latency_ms=0.0, fallback=False, detail=None):
        result = LLMResult(request_id=request_id, task=task, provider=provider,
                           model=model, latency_ms=latency_ms, fallback=fallback,
                           failure={"code": code, "message": str(message)[:300]})
        if detail:
            result.failure["detail"] = detail
        self._publish(result)
        return result

    def _publish(self, result):
        if self.bus is None:
            return
        payload = {
            "request_id": result.request_id,
            "task": result.task,
            "provider": result.provider or None,
            "model": result.model or None,
            "fallback": bool(result.fallback),
            "ok": result.ok,
            "latency_ms": result.latency_ms,
            "usage": dict(result.usage or {}),
            "privacy": "redacted",
            "ts": time.time(),
        }
        if result.failure:
            payload["error"] = {k: v for k, v in result.failure.items()
                                 if k in {"code", "message"}}
        try:
            self.bus.publish(STATUS_TOPIC, payload)
        except Exception:
            # Telemetry must never turn an otherwise usable provider response
            # into a failed robot behavior.
            pass
