import asyncio
import json
from types import SimpleNamespace

from omniagent.agents.state import AgentState
from omniagent.gateway.api import APIContext, get_health
from omniagent.gateway.errors import PUBLIC_ERROR_FALLBACK, public_error_message
from omniagent.gateway.router import IncomingMessage, MessageRouter


class FakeRequest:
    def __init__(self, ctx):
        self.app = {"api_ctx": ctx}


def test_public_error_message_unwraps_json_and_uses_vercel_agent_name():
    raw_error = json.dumps(
        {
            "error": (
                "This Vercel team requires SAML re-authentication before "
                "Omniagent can access team-scoped resources."
            ),
            "debug": "internal details",
        }
    )

    assert public_error_message(raw_error) == (
        "This Vercel team requires SAML re-authentication before "
        "Vercel Agent can access team-scoped resources."
    )


def test_public_error_message_hides_serialized_json_without_display_message():
    raw_error = json.dumps({"debug": {"reason": "internal_error"}})

    assert public_error_message(raw_error) == PUBLIC_ERROR_FALLBACK


def test_public_error_message_redacts_private_values():
    assert public_error_message("Upstream failed with token=secret-token") == (
        "Upstream failed with token=****"
    )
    assert public_error_message("Authorization: Bearer secret-token") == (
        "Authorization: Bearer ****"
    )


def test_router_does_not_send_raw_exception_details_to_channels():
    async def run():
        router = MessageRouter()

        async def handler(_message):
            raise RuntimeError(
                json.dumps(
                    {
                        "error": (
                            "This Vercel team requires SAML re-authentication before "
                            "OmniAgent can access team-scoped resources."
                        ),
                        "debug": "internal details",
                    }
                )
            )

        router.set_agent_handler(handler)

        outgoing = await router.route_to_agent(
            IncomingMessage(
                session_id="session_123",
                user_id="user_123",
                channel_id="slack",
                content="hello",
                metadata={},
            )
        )

        assert outgoing.content == (
            "Error: This Vercel team requires SAML re-authentication before "
            "Vercel Agent can access team-scoped resources."
        )
        assert "debug" not in outgoing.content
        assert "{" not in outgoing.content

    asyncio.run(run())


def test_health_sanitizes_agent_error():
    async def run():
        state = AgentState()
        state.error = json.dumps(
            {
                "error": (
                    "This Vercel team requires SAML re-authentication before "
                    "Omniagent can access team-scoped resources."
                ),
                "debug": "internal details",
            }
        )
        agent = SimpleNamespace(
            agent_state=state,
            config=SimpleNamespace(
                agent=SimpleNamespace(model_provider="deepseek", model_id="deepseek-chat")
            ),
            memory_manager=None,
            policy=None,
            approval_manager=None,
        )
        ctx = APIContext(agent=agent, session_manager=None, config=None, channel_manager=None)
        response = await get_health(FakeRequest(ctx))

        data = json.loads(response.text)
        assert data["agent"]["error"] == (
            "This Vercel team requires SAML re-authentication before "
            "Vercel Agent can access team-scoped resources."
        )

    asyncio.run(run())
