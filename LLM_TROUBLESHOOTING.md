# LLM gateway troubleshooting

All production calls from conversation, intent repair, object identification,
coaching, and reflection pass through `layer_b/llm_gateway.py`. The gateway
does not log prompts, images, transcripts, API keys, or tool arguments.

Watch provider metadata and the service using the production service names:

```bash
mosquitto_sub -v -t picarx/llm/status
sudo journalctl -u picarx-orchestrator.service -f -o cat
```

Each status record identifies the request task, selected provider/model,
latency, normalized usage, fallback flag, and a bounded error code. A
`policy_refusal` is not retried through another provider. Quota, timeout,
transient, or missing-optional-SDK failures can use OpenAI only for requests
explicitly marked idempotent; otherwise the caller returns to its previous
cache-only or apology behavior.

Check only whether credentials are present, never print their values:

```bash
systemctl show picarx-orchestrator.service -p Environment
test -n "$ANTHROPIC_API_KEY" && echo 'Anthropic key present' || echo 'Anthropic key absent'
test -n "$OPENAI_API_KEY" && echo 'OpenAI key present' || echo 'OpenAI key absent'
```

The model IDs are configurable under the `llm` section of
`layer_b/config.json` or through the registered `LLM_*_MODEL` environment
variables. Missing keys, SDKs, malformed responses, privacy blocks, and
telemetry failures are fail-soft; they must not stop `picarx-safety.service`
or the orchestrator.
