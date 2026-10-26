"""Google ADK agent workflow with Model Armor prompt screening, Vertex AI Search, and Cloud SDP."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, AsyncGenerator, Callable, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import VertexAiSearchTool
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from config.settings import AgentSettings, get_settings, is_placeholder_value
from .sdp_service import (
    PromptGuardrailService,
    RedactionResult,
    SDPService,
)

logger = logging.getLogger(__name__)

APP_NAME = "enterprise_sdp_capable_agent"

SYSTEM_INSTRUCTION = """You are an enterprise knowledge assistant deployed in Gemini Enterprise.
When a Discovery Engine Data Store tool is available, search the connected enterprise repository
before answering questions about internal documents, policies, agreements, audits, or records.
Provide accurate, concise, well-structured answers grounded in retrieved source documents without fabricating citations."""


class ResilientVertexGemini(Gemini):
    """ADK Gemini wrapper pinned to configured Vertex location with optional model fallback."""

    fallback_model: Optional[str] = None

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        try:
            async for resp in super().generate_content_async(
                llm_request=llm_request, stream=stream
            ):
                yield resp
        except genai_errors.ClientError as exc:
            if (
                exc.code == 404
                and self.fallback_model
                and self.model != self.fallback_model
            ):
                logger.warning(
                    "Vertex AI model '%s' returned 404 NOT_FOUND; falling back to '%s'.",
                    self.model,
                    self.fallback_model,
                )
                self.model = self.fallback_model
                llm_request.model = self.fallback_model
                async for resp in super().generate_content_async(
                    llm_request=llm_request, stream=stream
                ):
                    yield resp
            else:
                raise


class AgentTurnResponse(BaseModel):
    """Structured result returned from an end-to-end ADK agent turn."""

    response: str
    user_id: str
    session_id: str
    prompt_blocked: bool = False
    blocked_reason: Optional[str] = None
    redacted: bool = False
    transformed_count: int = 0
    applied_info_types: list[str] = Field(default_factory=list)
    templates_applied: dict[str, Optional[str]] = Field(default_factory=dict)


def configure_vertex_environment(settings: AgentSettings) -> None:
    """Configure google-genai / ADK environment defaults without clobbering Agent Engine."""
    os.environ.setdefault("GOOGLE_GENAI_USE_ENTERPRISE", "TRUE")
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    if not is_placeholder_value(settings.project_id):
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.project_id)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.vertex_location)


def build_before_model_callback(
    guardrail_service: PromptGuardrailService,
) -> Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]:
    """Create ADK before_model_callback enforcing Model Armor binary prompt screening.

    Note: ADK invokes callbacks using keyword arguments (`callback_context`, `llm_request`),
    so parameter names must match exactly.
    """

    def before_model_callback(
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        callback_context.state["sdp_redacted"] = False
        callback_context.state["sdp_transformed_count"] = 0
        callback_context.state["sdp_info_types"] = []

        latest_user_content: Optional[types.Content] = None
        if llm_request.contents:
            for content in reversed(llm_request.contents):
                if content.role == "user" and content.parts:
                    latest_user_content = content
                    break

        if latest_user_content is None or not latest_user_content.parts:
            callback_context.state["prompt_blocked"] = False
            return None

        combined_user_text = "\n".join(
            part.text for part in latest_user_content.parts if part.text
        )
        screening = guardrail_service.screen_prompt(combined_user_text)

        callback_context.state["prompt_blocked"] = not screening.allowed
        callback_context.state["blocked_reason"] = screening.blocked_reason
        callback_context.state["prompt_template"] = screening.prompt_filter_template

        if not screening.allowed:
            refusal_text = (
                "Request declined by enterprise guardrail policy "
                f"({screening.prompt_filter_template}): "
                "The prompt triggered a binary safety block and was not sent to the model."
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=refusal_text)],
                )
            )

        for part in latest_user_content.parts:
            if part.text:
                part.text = part.text.strip()

        return None

    return before_model_callback


def build_after_model_callback(
    sdp_service: SDPService,
) -> Callable[[CallbackContext, LlmResponse], Optional[LlmResponse]]:
    """Create ADK after_model_callback applying in-place Cloud SDP redaction.

    Note: ADK invokes callbacks using keyword arguments (`callback_context`, `llm_response`),
    so parameter names must match exactly.
    """

    def after_model_callback(
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        if getattr(llm_response, "partial", False):
            return None

        if not llm_response.content or not llm_response.content.parts:
            return None

        new_parts: list[types.Part] = []
        any_modified = False
        total_transformed = 0
        detected_info_types: list[str] = []

        for part in llm_response.content.parts:
            if not part.text or getattr(part, "thought", False):
                new_parts.append(part)
                continue

            redaction: RedactionResult = sdp_service.deidentify_with_metadata(part.text)
            if redaction.was_modified:
                any_modified = True
                total_transformed += redaction.transformed_count
                for info_type in redaction.applied_info_types:
                    if info_type not in detected_info_types:
                        detected_info_types.append(info_type)
                new_parts.append(types.Part(text=redaction.redacted_text))
            else:
                new_parts.append(part)

        callback_context.state["sdp_redacted"] = any_modified
        callback_context.state["sdp_transformed_count"] = total_transformed
        callback_context.state["sdp_info_types"] = detected_info_types
        callback_context.state["active_inspect_template"] = (
            sdp_service.inspect_template_path
        )

        if not any_modified:
            return None

        redacted_content = types.Content(
            role=llm_response.content.role or "model",
            parts=new_parts,
        )
        return LlmResponse(
            content=redacted_content,
            grounding_metadata=getattr(llm_response, "grounding_metadata", None),
            custom_metadata=getattr(llm_response, "custom_metadata", None),
            usage_metadata=getattr(llm_response, "usage_metadata", None),
        )

    return after_model_callback


def create_sdp_agent(
    settings: Optional[AgentSettings] = None,
    sdp_service: Optional[SDPService] = None,
    guardrail_service: Optional[PromptGuardrailService] = None,
    model_override: Optional[str | BaseLlm] = None,
) -> LlmAgent:
    """Instantiate the ADK LlmAgent wired with Discovery Engine Search + SDP callbacks."""
    resolved_settings = settings or get_settings()
    configure_vertex_environment(resolved_settings)

    resolved_sdp = sdp_service or SDPService(
        project_id=resolved_settings.project_id,
        location=resolved_settings.location,
        inspect_template_path=resolved_settings.inspect_template_path,
        fallback_inspect_template_path=resolved_settings.fallback_inspect_template_path,
        deidentify_template_path=resolved_settings.deidentify_template_path,
        include_supplemental_inspect_config=resolved_settings.enable_supplemental_inspect_config,
    )
    resolved_guardrail = guardrail_service or PromptGuardrailService(
        project_id=resolved_settings.project_id,
        location=resolved_settings.location,
        template_id=resolved_settings.prompt_filter_template_id,
        template_path=resolved_settings.prompt_filter_template_path,
        enable_model_armor_api=resolved_settings.enable_model_armor_api,
    )

    # Explicitly pin genai.Client to `vertex_location` (e.g., `global`) so running inside
    # a regional Agent Engine container (e.g., `us-west1`) never overrides model/grounding endpoints.
    resolved_model: str | BaseLlm = model_override or ResilientVertexGemini(
        model=resolved_settings.gemini_model,
        fallback_model=resolved_settings.fallback_gemini_model,
        client_kwargs={
            "vertexai": True,
            "project": resolved_settings.project_id,
            "location": resolved_settings.vertex_location,
        },
    )

    tools: list[Any] = []
    if (
        model_override is None
        and resolved_settings.enable_datastore_search
        and resolved_settings.data_store_id
        and not is_placeholder_value(resolved_settings.data_store_id)
    ):
        tools.append(
            VertexAiSearchTool(
                data_store_id=resolved_settings.data_store_id,
                bypass_multi_tools_limit=False,
            )
        )

    return LlmAgent(
        name="enterprise_sdp_capable_agent",
        model=resolved_model,
        description=(
            "Enterprise knowledge assistant grounded on Discovery Engine Data Stores "
            "with Model Armor prompt screening and Cloud SDP post-inference redaction."
        ),
        instruction=SYSTEM_INSTRUCTION,
        tools=tools,
        before_model_callback=build_before_model_callback(resolved_guardrail),
        after_model_callback=build_after_model_callback(resolved_sdp),
    )


class SdpAgentWorkflow:
    """High-level stateful ADK Runner orchestrator for FastAPI, CLI, and test suites."""

    def __init__(
        self,
        settings: Optional[AgentSettings] = None,
        sdp_service: Optional[SDPService] = None,
        guardrail_service: Optional[PromptGuardrailService] = None,
        model_override: Optional[str | BaseLlm] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.sdp_service = sdp_service or SDPService(
            project_id=self.settings.project_id,
            location=self.settings.location,
            inspect_template_path=self.settings.inspect_template_path,
            fallback_inspect_template_path=self.settings.fallback_inspect_template_path,
            deidentify_template_path=self.settings.deidentify_template_path,
            include_supplemental_inspect_config=self.settings.enable_supplemental_inspect_config,
        )
        self.guardrail_service = guardrail_service or PromptGuardrailService(
            project_id=self.settings.project_id,
            location=self.settings.location,
            template_id=self.settings.prompt_filter_template_id,
            template_path=self.settings.prompt_filter_template_path,
            enable_model_armor_api=self.settings.enable_model_armor_api,
        )
        self.agent = create_sdp_agent(
            settings=self.settings,
            sdp_service=self.sdp_service,
            guardrail_service=self.guardrail_service,
            model_override=model_override,
        )
        self.runner = InMemoryRunner(agent=self.agent, app_name=APP_NAME)
        self._known_sessions: set[tuple[str, str]] = set()

    async def ensure_session(self, user_id: str, session_id: str) -> None:
        """Create an ADK session if one does not already exist."""
        key = (user_id, session_id)
        if key in self._known_sessions:
            return
        existing = await self.runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if existing is None:
            await self.runner.session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
        self._known_sessions.add(key)

    async def chat(
        self,
        prompt: str,
        user_id: str = "enterprise-user",
        session_id: Optional[str] = None,
    ) -> AgentTurnResponse:
        """Execute a full ADK turn: prompt check -> Vertex Gemini + Search -> SDP redaction."""
        resolved_session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        await self.ensure_session(user_id=user_id, session_id=resolved_session_id)

        user_message = types.Content(
            role="user",
            parts=[types.Part(text=prompt)],
        )

        final_chunks: list[str] = []
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=resolved_session_id,
            new_message=user_message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_chunks.append(part.text)

        session = await self.runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=resolved_session_id,
        )
        state: dict[str, Any] = dict(session.state) if session and session.state else {}

        return AgentTurnResponse(
            response="\n".join(final_chunks).strip(),
            user_id=user_id,
            session_id=resolved_session_id,
            prompt_blocked=bool(state.get("prompt_blocked", False)),
            blocked_reason=state.get("blocked_reason"),
            redacted=bool(state.get("sdp_redacted", False)),
            transformed_count=int(state.get("sdp_transformed_count", 0)),
            applied_info_types=list(state.get("sdp_info_types", [])),
            templates_applied={
                "prompt_filter": self.settings.prompt_filter_template_id,
                "response_filter": self.settings.response_filter_template_id,
                "inspect_template": state.get(
                    "active_inspect_template",
                    self.settings.inspect_template_path,
                ),
                "deidentify_template": self.settings.deidentify_template_path,
                "data_store": self.settings.data_store_id,
            },
        )


# Standard ADK module-level entrypoint for `adk web` and `adk deploy agent_engine`
root_agent: LlmAgent = create_sdp_agent()
