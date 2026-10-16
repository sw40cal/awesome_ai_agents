"""Minimal Claude agent that uses Perplexity AI tools through a Composio session."""

import os
import sys

from anthropic import Anthropic
from composio import Composio
from composio_anthropic import AnthropicProvider
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 4096
MAX_TURNS = 10
TOOLKITS = ["perplexityai"]


def create_session(composio, user_id):
    """Create a Composio session for one user with only Perplexity AI enabled."""
    return composio.create(user_id=user_id, toolkits=TOOLKITS)


def run_agent(prompt, *, client, provider, session, model=DEFAULT_MODEL):
    """Run a tool-use loop until Claude stops calling tools; return its final text."""
    tools = session.tools()
    messages = [{"role": "user", "content": prompt}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            return "".join(b.text for b in response.content if b.type == "text")

        results = provider.handle_tool_calls(session=session, response=response)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(result),
                        "is_error": not result.get("successful", False),
                    }
                    for tool_use, result in zip(tool_uses, results)
                ],
            }
        )

    raise RuntimeError(f"Agent did not finish within {MAX_TURNS} turns")


def main():
    load_dotenv()
    user_id = os.environ.get("COMPOSIO_USER_ID")
    if not user_id:
        sys.exit("Set COMPOSIO_USER_ID to the user this agent acts for.")

    prompt = " ".join(sys.argv[1:]) or (
        "Use Perplexity to find one notable AI agent framework release this month. "
        "Answer in two sentences and cite the source."
    )
    provider = AnthropicProvider()
    composio = Composio(provider=provider)  # reads COMPOSIO_API_KEY
    session = create_session(composio, user_id)
    print(
        run_agent(
            prompt,
            client=Anthropic(),  # reads ANTHROPIC_API_KEY
            provider=provider,
            session=session,
            model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        )
    )


if __name__ == "__main__":
    main()
