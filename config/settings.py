"""Environment-driven settings for Google ADK, Model Armor, Discovery Engine, and Cloud SDP."""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is optional at runtime
    pass

# Default placeholder constants (override via environment variables or .env file)
PROJECT_ID: str = "your-gcp-project-id"
PROJECT_NUMBER: str = "123456789012"
LOCATION: str = "us"
AGENT_ENGINE_REGION: str = "us-west1"
VERTEX_LOCATION: str = "global"
GEMINI_MODEL: str = "gemini-2.5-flash"
FALLBACK_GEMINI_MODEL: str = "gemini-3.5-flash"

PROMPT_FILTER_TEMPLATE_ID: str = "your-prompt-filter-template"
RESPONSE_FILTER_TEMPLATE_ID: str = "your-response-filter-template"

PROMPT_FILTER_TEMPLATE_PATH: str = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{PROMPT_FILTER_TEMPLATE_ID}"
)
RESPONSE_FILTER_TEMPLATE_PATH: str = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{RESPONSE_FILTER_TEMPLATE_ID}"
)

INSPECT_TEMPLATE_PATH: str = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/inspectTemplates/your-inspect-template"
)
DEIDENTIFY_TEMPLATE_PATH: str = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/deidentifyTemplates/your-deidentify-template"
)

GEMINI_ENTERPRISE_ENGINE_ID: str = "your-gemini-enterprise-engine-id"
DATA_STORE_ID: str = (
    f"projects/{PROJECT_NUMBER}/locations/global/collections/default_collection/"
    "dataStores/your-data-store-id"
)

_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "your-gcp-project-id",
    "123456789012",
    "your-prompt-filter-template",
    "your-response-filter-template",
    "your-inspect-template",
    "your-deidentify-template",
    "your-gemini-enterprise-engine-id",
    "your-data-store-id",
    "<YOUR_",
)


def is_placeholder_value(value: Optional[str]) -> bool:
    """Return True if a setting is empty or still contains a template placeholder."""
    if not value or not value.strip():
        return True
    return any(marker in value for marker in _PLACEHOLDER_MARKERS)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class AgentSettings(BaseModel):
    """Typed runtime configuration for the ADK SDP-capable agent."""

    project_id: str = Field(default=PROJECT_ID)
    project_number: str = Field(default=PROJECT_NUMBER)
    location: str = Field(default=LOCATION)
    agent_engine_region: str = Field(default=AGENT_ENGINE_REGION)
    vertex_location: str = Field(default=VERTEX_LOCATION)
    gemini_model: str = Field(default=GEMINI_MODEL)
    fallback_gemini_model: Optional[str] = Field(default=FALLBACK_GEMINI_MODEL)

    inspect_template_path: Optional[str] = Field(default=INSPECT_TEMPLATE_PATH)
    fallback_inspect_template_path: Optional[str] = Field(default=None)
    deidentify_template_path: str = Field(default=DEIDENTIFY_TEMPLATE_PATH)

    prompt_filter_template_id: str = Field(default=PROMPT_FILTER_TEMPLATE_ID)
    response_filter_template_id: str = Field(default=RESPONSE_FILTER_TEMPLATE_ID)
    prompt_filter_template_path: str = Field(default=PROMPT_FILTER_TEMPLATE_PATH)
    response_filter_template_path: str = Field(default=RESPONSE_FILTER_TEMPLATE_PATH)

    gemini_enterprise_engine_id: str = Field(default=GEMINI_ENTERPRISE_ENGINE_ID)
    data_store_id: Optional[str] = Field(default=DATA_STORE_ID)
    enable_datastore_search: bool = Field(default=True)
    enable_model_armor_api: bool = Field(default=True)
    enable_supplemental_inspect_config: bool = Field(default=True)

    @property
    def sdp_parent(self) -> str:
        """Parent resource string used by Cloud DLP / SDP API calls."""
        return f"projects/{self.project_id}/locations/{self.location}"


def get_settings() -> AgentSettings:
    """Load runtime settings from environment variables with clean fallback defaults."""
    project_id = os.getenv("GCP_PROJECT_ID", PROJECT_ID)
    project_number = os.getenv("GCP_PROJECT_NUMBER", PROJECT_NUMBER)
    location = os.getenv("SDP_LOCATION", LOCATION)

    prompt_template_id = os.getenv(
        "MODEL_ARMOR_PROMPT_TEMPLATE_ID", PROMPT_FILTER_TEMPLATE_ID
    )
    response_template_id = os.getenv(
        "MODEL_ARMOR_RESPONSE_TEMPLATE_ID", RESPONSE_FILTER_TEMPLATE_ID
    )

    default_inspect_path = (
        f"projects/{project_id}/locations/{location}/inspectTemplates/your-inspect-template"
    )
    default_deid_path = (
        f"projects/{project_id}/locations/{location}/deidentifyTemplates/your-deidentify-template"
    )
    default_datastore = (
        f"projects/{project_number}/locations/global/collections/default_collection/"
        "dataStores/your-data-store-id"
    )

    return AgentSettings(
        project_id=project_id,
        project_number=project_number,
        location=location,
        agent_engine_region=os.getenv("AGENT_ENGINE_REGION", AGENT_ENGINE_REGION),
        vertex_location=os.getenv("VERTEX_LOCATION", VERTEX_LOCATION),
        gemini_model=os.getenv("GEMINI_MODEL", GEMINI_MODEL),
        fallback_gemini_model=os.getenv("FALLBACK_GEMINI_MODEL", FALLBACK_GEMINI_MODEL)
        or None,
        inspect_template_path=os.getenv("SDP_INSPECT_TEMPLATE_PATH", default_inspect_path),
        fallback_inspect_template_path=os.getenv("SDP_FALLBACK_INSPECT_TEMPLATE_PATH")
        or None,
        deidentify_template_path=os.getenv(
            "SDP_DEIDENTIFY_TEMPLATE_PATH", default_deid_path
        ),
        prompt_filter_template_id=prompt_template_id,
        response_filter_template_id=response_template_id,
        prompt_filter_template_path=os.getenv(
            "MODEL_ARMOR_PROMPT_TEMPLATE_PATH",
            f"projects/{project_id}/locations/{location}/templates/{prompt_template_id}",
        ),
        response_filter_template_path=os.getenv(
            "MODEL_ARMOR_RESPONSE_TEMPLATE_PATH",
            f"projects/{project_id}/locations/{location}/templates/{response_template_id}",
        ),
        gemini_enterprise_engine_id=os.getenv(
            "GEMINI_ENTERPRISE_ENGINE_ID", GEMINI_ENTERPRISE_ENGINE_ID
        ),
        data_store_id=os.getenv("DISCOVERY_ENGINE_DATA_STORE_ID", default_datastore),
        enable_datastore_search=_env_bool("ENABLE_DATASTORE_SEARCH", True),
        enable_model_armor_api=_env_bool("ENABLE_MODEL_ARMOR_API", True),
        enable_supplemental_inspect_config=_env_bool(
            "ENABLE_SUPPLEMENTAL_INSPECT_CONFIG", True
        ),
    )
