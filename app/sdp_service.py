"""Cloud Sensitive Data Protection (SDP / DLP v2) and Model Armor middleware services."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Protocol

from google.api_core import exceptions as gcp_exceptions
from google.cloud import dlp_v2
from pydantic import BaseModel, Field

from config.settings import (
    DEIDENTIFY_TEMPLATE_PATH,
    INSPECT_TEMPLATE_PATH,
    LOCATION,
    PROJECT_ID,
    PROMPT_FILTER_TEMPLATE_ID,
    PROMPT_FILTER_TEMPLATE_PATH,
    RESPONSE_FILTER_TEMPLATE_ID,
    is_placeholder_value,
)

logger = logging.getLogger(__name__)


class DlpClientProtocol(Protocol):
    """Protocol matching google.cloud.dlp_v2.DlpServiceClient methods used by SDPService."""

    def deidentify_content(
        self,
        request: Optional[dlp_v2.DeidentifyContentRequest | dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dlp_v2.DeidentifyContentResponse: ...

    def inspect_content(
        self,
        request: Optional[dlp_v2.InspectContentRequest | dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dlp_v2.InspectContentResponse: ...


class RedactionResult(BaseModel):
    """Structured metadata returned after SDP de-identification."""

    original_length: int
    redacted_text: str
    was_modified: bool
    transformed_count: int = 0
    applied_info_types: list[str] = Field(default_factory=list)
    deidentify_template_path: str = DEIDENTIFY_TEMPLATE_PATH
    inspect_template_path: Optional[str] = INSPECT_TEMPLATE_PATH
    response_filter_template: str = RESPONSE_FILTER_TEMPLATE_ID


class PromptSanitizationResult(BaseModel):
    """Outcome of pre-inference prompt screening (Model Armor binary block + sanitization)."""

    allowed: bool
    sanitized_prompt: str
    blocked_reason: Optional[str] = None
    prompt_filter_template: str = PROMPT_FILTER_TEMPLATE_ID


class SDPService:
    """Helper wrapping Google Cloud DlpServiceClient for template-driven SDP redaction."""

    # Optional supplemental detectors merged alongside the inspect template so synthetic
    # test SSNs (e.g., SSA-reserved 000-00-0000), test cards, emails, and API keys are
    # consistently caught during staging/UAT tests as well as production traffic.
    _SUPPLEMENTAL_INFO_TYPES: tuple[str, ...] = (
        "EMAIL_ADDRESS",
        "GCP_API_KEY",
        "US_SOCIAL_SECURITY_NUMBER",
        "CREDIT_CARD_NUMBER",
    )

    def __init__(
        self,
        project_id: str = PROJECT_ID,
        location: str = LOCATION,
        inspect_template_path: Optional[str] = INSPECT_TEMPLATE_PATH,
        fallback_inspect_template_path: Optional[str] = None,
        deidentify_template_path: str = DEIDENTIFY_TEMPLATE_PATH,
        include_supplemental_inspect_config: bool = True,
        dlp_client: Optional[DlpClientProtocol] = None,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.inspect_template_path = inspect_template_path
        self.fallback_inspect_template_path = fallback_inspect_template_path
        self.deidentify_template_path = deidentify_template_path
        self.include_supplemental_inspect_config = include_supplemental_inspect_config
        self._dlp_client = dlp_client

    @property
    def parent(self) -> str:
        """Standard Cloud SDP regional/multi-regional parent resource path."""
        return f"projects/{self.project_id}/locations/{self.location}"

    @property
    def client(self) -> DlpClientProtocol:
        """Lazily instantiate the official Google Cloud DlpServiceClient."""
        if self._dlp_client is None:
            self._dlp_client = dlp_v2.DlpServiceClient()
        return self._dlp_client

    def build_supplemental_inspect_config(self) -> dlp_v2.InspectConfig:
        """Build supplemental InspectConfig merged alongside the SDP InspectTemplate."""
        return dlp_v2.InspectConfig(
            info_types=[
                dlp_v2.InfoType(name=name) for name in self._SUPPLEMENTAL_INFO_TYPES
            ],
            custom_info_types=[
                dlp_v2.CustomInfoType(
                    info_type=dlp_v2.InfoType(name="US_SOCIAL_SECURITY_NUMBER"),
                    regex=dlp_v2.CustomInfoType.Regex(
                        pattern=r"\b\d{3}-\d{2}-\d{4}\b"
                    ),
                    likelihood=dlp_v2.Likelihood.VERY_LIKELY,
                ),
                dlp_v2.CustomInfoType(
                    info_type=dlp_v2.InfoType(name="CREDIT_CARD_NUMBER"),
                    regex=dlp_v2.CustomInfoType.Regex(
                        pattern=r"\b(?:4[0-9]{3}|5[1-5][0-9]{2}|6011)[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b"
                    ),
                    likelihood=dlp_v2.Likelihood.VERY_LIKELY,
                ),
                dlp_v2.CustomInfoType(
                    info_type=dlp_v2.InfoType(name="GCP_API_KEY"),
                    regex=dlp_v2.CustomInfoType.Regex(
                        pattern=r"\bAIza[0-9A-Za-z\-_]{31,45}\b"
                    ),
                    likelihood=dlp_v2.Likelihood.VERY_LIKELY,
                ),
            ],
            min_likelihood=dlp_v2.Likelihood.POSSIBLE,
        )

    def build_deidentify_request(
        self,
        text: str,
        inspect_template_override: Optional[str] = None,
    ) -> dlp_v2.DeidentifyContentRequest:
        """Construct a protobuf DeidentifyContentRequest using configured SDP template paths."""
        item = dlp_v2.ContentItem(value=text)
        active_inspect_template = (
            inspect_template_override
            if inspect_template_override is not None
            else self.inspect_template_path
        )
        request_kwargs: dict[str, Any] = {
            "parent": self.parent,
            "deidentify_template_name": self.deidentify_template_path,
            "item": item,
        }
        if self.include_supplemental_inspect_config:
            request_kwargs["inspect_config"] = self.build_supplemental_inspect_config()
        if active_inspect_template:
            request_kwargs["inspect_template_name"] = active_inspect_template

        return dlp_v2.DeidentifyContentRequest(**request_kwargs)

    def deidentify_with_metadata(self, text: str) -> RedactionResult:
        """Inspect and redact sensitive spans in-place without blocking surrounding text."""
        if not text or not text.strip():
            return RedactionResult(
                original_length=len(text),
                redacted_text=text,
                was_modified=False,
                deidentify_template_path=self.deidentify_template_path,
                inspect_template_path=self.inspect_template_path,
            )

        request = self.build_deidentify_request(text)
        applied_inspect_template = self.inspect_template_path

        try:
            response: dlp_v2.DeidentifyContentResponse = (
                self.client.deidentify_content(request=request)
            )
        except gcp_exceptions.NotFound as exc:
            if (
                self.fallback_inspect_template_path
                and self.fallback_inspect_template_path != self.inspect_template_path
                and "inspect template" in str(exc).lower()
            ):
                logger.warning(
                    "Inspect template '%s' returned 404; retrying with fallback '%s'.",
                    self.inspect_template_path,
                    self.fallback_inspect_template_path,
                )
                self.inspect_template_path = self.fallback_inspect_template_path
                applied_inspect_template = self.fallback_inspect_template_path
                retry_request = self.build_deidentify_request(
                    text,
                    inspect_template_override=self.fallback_inspect_template_path,
                )
                response = self.client.deidentify_content(request=retry_request)
            else:
                raise

        redacted_text = response.item.value
        applied_info_types: list[str] = []
        transformed_count = 0

        if response.overview and response.overview.transformation_summaries:
            for summary in response.overview.transformation_summaries:
                if summary.info_type and summary.info_type.name:
                    if summary.info_type.name not in applied_info_types:
                        applied_info_types.append(summary.info_type.name)
                for result in summary.results:
                    transformed_count += int(result.count or 0)

        return RedactionResult(
            original_length=len(text),
            redacted_text=redacted_text,
            was_modified=(redacted_text != text),
            transformed_count=transformed_count,
            applied_info_types=applied_info_types,
            deidentify_template_path=self.deidentify_template_path,
            inspect_template_path=applied_inspect_template,
        )

    def deidentify_text(self, text: str) -> str:
        """Convenience wrapper returning only the redacted text."""
        return self.deidentify_with_metadata(text).redacted_text


class PromptGuardrailService:
    """Pre-inference binary prompt filter combining local guardrails with Cloud Model Armor."""

    _BLOCK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "prompt_injection_override",
            re.compile(
                r"(ignore\s+(all\s+)?(previous|prior|system)\s+instructions|"
                r"bypass\s+(all\s+)?(safety|guardrails|model\s+armor|dlp|sdp)|"
                r"reveal\s+(your\s+)?system\s+prompt)",
                re.IGNORECASE,
            ),
        ),
        (
            "exfiltrate_raw_pii",
            re.compile(
                r"(dump\s+all\s+(unredacted\s+)?(ssns?|social\s+security\s+numbers?|credit\s+cards?)|"
                r"exfiltrate\s+(client|confidential|privileged|customer)\s+records)",
                re.IGNORECASE,
            ),
        ),
    )

    def __init__(
        self,
        project_id: str = PROJECT_ID,
        location: str = LOCATION,
        template_id: str = PROMPT_FILTER_TEMPLATE_ID,
        template_path: str = PROMPT_FILTER_TEMPLATE_PATH,
        enable_model_armor_api: bool = False,
        model_armor_client: Any = None,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.template_id = template_id
        self.template_path = template_path
        self.enable_model_armor_api = enable_model_armor_api
        self._model_armor_client = model_armor_client

    def screen_prompt(self, prompt: str) -> PromptSanitizationResult:
        """Screen incoming user prompt; return binary allow/block + normalized text."""
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", prompt).strip()

        for rule_name, pattern in self._BLOCK_PATTERNS:
            if pattern.search(normalized):
                logger.warning(
                    "Prompt blocked by %s (rule=%s)",
                    self.template_id,
                    rule_name,
                )
                return PromptSanitizationResult(
                    allowed=False,
                    sanitized_prompt=normalized,
                    blocked_reason=(
                        f"Blocked by Model Armor policy '{self.template_id}' "
                        f"({rule_name})."
                    ),
                    prompt_filter_template=self.template_id,
                )

        if self.enable_model_armor_api and not is_placeholder_value(self.template_path):
            api_block_reason = self._check_model_armor_api(normalized)
            if api_block_reason:
                return PromptSanitizationResult(
                    allowed=False,
                    sanitized_prompt=normalized,
                    blocked_reason=api_block_reason,
                    prompt_filter_template=self.template_id,
                )

        return PromptSanitizationResult(
            allowed=True,
            sanitized_prompt=normalized,
            prompt_filter_template=self.template_id,
        )

    def _check_model_armor_api(self, prompt: str) -> Optional[str]:
        """Invoke Cloud Model Armor sanitize_user_prompt against the configured template."""
        try:
            from google.cloud import modelarmor_v1

            if self._model_armor_client is None:
                endpoint = (
                    f"modelarmor.{self.location}.rep.googleapis.com"
                    if self.location != "global"
                    else "modelarmor.googleapis.com"
                )
                self._model_armor_client = modelarmor_v1.ModelArmorClient(
                    client_options={"api_endpoint": endpoint}
                )

            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=self.template_path,
                user_prompt_data=modelarmor_v1.DataItem(text=prompt),
            )
            response = self._model_armor_client.sanitize_user_prompt(request=request)
            match_state = getattr(
                response.sanitization_result, "filter_match_state", None
            )
            if (
                match_state is not None
                and int(match_state)
                == int(modelarmor_v1.FilterMatchState.MATCH_FOUND)
            ):
                return f"Blocked by Cloud Model Armor template '{self.template_path}'."
        except Exception as exc:  # pragma: no cover - graceful degradation if offline
            logger.warning("Model Armor API check skipped/degraded: %s", exc)
        return None


class TemplateSimulatingDlpClient:
    """Protobuf-faithful DlpServiceClient simulator for offline unit and CLI testing."""

    _SIMULATED_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "US_SOCIAL_SECURITY_NUMBER",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        ),
        (
            "GCP_API_KEY",
            re.compile(r"\bAIza[0-9A-Za-z\-_]{20,45}\b"),
        ),
        (
            "CREDIT_CARD_NUMBER",
            re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        ),
        (
            "EMAIL_ADDRESS",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ),
        (
            "US_PHONE_NUMBER",
            re.compile(
                r"(?<!\d)(?:\+1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
            ),
        ),
    )

    def __init__(self) -> None:
        self.recorded_requests: list[dlp_v2.DeidentifyContentRequest] = []

    def deidentify_content(
        self,
        request: Optional[dlp_v2.DeidentifyContentRequest | dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dlp_v2.DeidentifyContentResponse:
        if isinstance(request, dict):
            req = dlp_v2.DeidentifyContentRequest(**request)
        elif isinstance(request, dlp_v2.DeidentifyContentRequest):
            req = request
        else:
            raise ValueError("Expected DeidentifyContentRequest protobuf or dict")

        if not req.parent:
            raise ValueError("DeidentifyContentRequest.parent is required")
        if not req.deidentify_template_name:
            raise ValueError(
                "DeidentifyContentRequest.deidentify_template_name is required"
            )

        self.recorded_requests.append(req)
        redacted = req.item.value
        summaries: list[dlp_v2.TransformationSummary] = []

        for info_type_name, pattern in self._SIMULATED_DETECTORS:
            matches = list(pattern.finditer(redacted))
            if matches:
                redacted = pattern.sub(f"[{info_type_name}]", redacted)
                summaries.append(
                    dlp_v2.TransformationSummary(
                        info_type=dlp_v2.InfoType(name=info_type_name),
                        results=[
                            dlp_v2.TransformationSummary.SummaryResult(
                                count=len(matches),
                                code=dlp_v2.TransformationResultStatusType.SUCCESS,
                            )
                        ],
                    )
                )

        return dlp_v2.DeidentifyContentResponse(
            item=dlp_v2.ContentItem(value=redacted),
            overview=dlp_v2.TransformationOverview(
                transformed_bytes=len(req.item.value.encode("utf-8")),
                transformation_summaries=summaries,
            ),
        )

    def inspect_content(
        self,
        request: Optional[dlp_v2.InspectContentRequest | dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dlp_v2.InspectContentResponse:
        if isinstance(request, dict):
            req = dlp_v2.InspectContentRequest(**request)
        elif isinstance(request, dlp_v2.InspectContentRequest):
            req = request
        else:
            raise ValueError("Expected InspectContentRequest protobuf or dict")

        findings: list[dlp_v2.Finding] = []
        text = req.item.value
        for info_type_name, pattern in self._SIMULATED_DETECTORS:
            for match in pattern.finditer(text):
                findings.append(
                    dlp_v2.Finding(
                        quote=match.group(0),
                        info_type=dlp_v2.InfoType(name=info_type_name),
                        likelihood=dlp_v2.Likelihood.VERY_LIKELY,
                    )
                )

        return dlp_v2.InspectContentResponse(
            result=dlp_v2.InspectResult(findings=findings)
        )
