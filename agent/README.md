# Perplexity agent (Composio + Claude)

A minimal Claude agent whose tools come from a Composio session scoped to the
`perplexityai` toolkit.

```bash
cd agent
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in COMPOSIO_API_KEY, ANTHROPIC_API_KEY, COMPOSIO_USER_ID
python agent.py "What's new in AI agents this week?"
```

If the user hasn't connected Perplexity yet, the agent's response includes a
Composio Connect Link; open it, authorize, and run again.

Tests: `pip install pytest && pytest tests`
