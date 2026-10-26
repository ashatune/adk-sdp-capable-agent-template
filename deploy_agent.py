#!/usr/bin/env python3
"""Deploy the ADK SDP-Capable Agent to Vertex AI Agent Engine and register in Gemini Enterprise."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

from config.settings import get_settings, is_placeholder_value

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DISPLAY_NAME = "Enterprise SDP Redaction Agent"
DEFAULT_DESCRIPTION = (
    "Google ADK enterprise assistant grounded on Discovery Engine Data Stores with "
    "Model Armor prompt screening and post-inference Cloud SDP in-place redaction."
)


def get_access_token() -> str:
    return (
        subprocess.check_output(["gcloud", "auth", "print-access-token"])
        .decode("utf-8")
        .strip()
    )


def resolve_adk_binary() -> str:
    venv_adk = REPO_ROOT / ".venv" / "bin" / "adk"
    if venv_adk.exists():
        return str(venv_adk)
    found = shutil.which("adk")
    if found:
        return found
    raise FileNotFoundError(
        "Could not find 'adk' CLI. Install dependencies with `pip install -r requirements.txt`."
    )


def find_existing_reasoning_engine(
    project_id: str,
    project_number: str,
    region: str,
    display_name: str,
) -> str | None:
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "X-Goog-User-Project": project_id,
    }
    url = (
        f"https://{region}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{project_number}/locations/{region}/reasoningEngines"
    )
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    for engine in resp.json().get("reasoningEngines", []):
        if engine.get("displayName") == display_name:
            return engine["name"]
    return None


def deploy_reasoning_engine(
    project_id: str,
    project_number: str,
    region: str,
    display_name: str,
    description: str,
    service_account: str | None,
) -> str:
    existing_name = find_existing_reasoning_engine(
        project_id, project_number, region, display_name
    )
    adk_bin = resolve_adk_binary()
    cmd = [
        adk_bin,
        "deploy",
        "agent_engine",
        f"--project={project_id}",
        f"--region={region}",
        f"--display_name={display_name}",
        f"--description={description}",
        f"--extra_packages={REPO_ROOT / 'config'}",
    ]
    if service_account and not is_placeholder_value(service_account):
        cmd.append(f"--service_account={service_account}")

    if existing_name:
        engine_id = existing_name.split("/")[-1]
        print(f"Updating existing Reasoning Engine: {existing_name}")
        cmd.append(f"--agent_engine_id={engine_id}")
    else:
        print(f"Creating new Reasoning Engine in {region}...")

    cmd.append(str(REPO_ROOT / "app"))
    subprocess.run(cmd, cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, check=True)

    deployed_name = find_existing_reasoning_engine(
        project_id, project_number, region, display_name
    )
    if not deployed_name:
        raise RuntimeError(
            "Reasoning Engine deployment succeeded, but lookup by displayName returned None."
        )
    print(f"Deployed Reasoning Engine: {deployed_name}")
    return deployed_name


def register_in_gemini_enterprise(
    project_id: str,
    project_number: str,
    ge_engine_id: str,
    reasoning_engine_name: str,
    display_name: str,
    description: str,
) -> dict:
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "X-Goog-User-Project": project_id,
        "Content-Type": "application/json",
    }
    agents_url = (
        "https://discoveryengine.googleapis.com/v1alpha/"
        f"projects/{project_number}/locations/global/collections/default_collection/"
        f"engines/{ge_engine_id}/assistants/default_assistant/agents"
    )
    payload = {
        "displayName": display_name,
        "description": description,
        "adkAgentDefinition": {
            "toolSettings": {
                "toolDescription": description,
            },
            "provisionedReasoningEngine": {
                "reasoningEngine": reasoning_engine_name,
            },
        },
    }

    existing_resp = requests.get(agents_url, headers=headers, timeout=30)
    existing_resp.raise_for_status()
    for agent in existing_resp.json().get("agents", []):
        if agent.get("displayName") == display_name:
            print(f"Updating existing Gemini Enterprise agent: {agent['name']}")
            patch_url = (
                f"https://discoveryengine.googleapis.com/v1alpha/{agent['name']}"
                "?updateMask=description,adkAgentDefinition"
            )
            upd = requests.patch(patch_url, headers=headers, json=payload, timeout=30)
            upd.raise_for_status()
            return upd.json()

    create_resp = requests.post(agents_url, headers=headers, json=payload, timeout=30)
    create_resp.raise_for_status()
    created = create_resp.json()
    print(f"Registered in Gemini Enterprise: {created.get('name')}")
    return created


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Deploy the ADK SDP-Capable Agent to Vertex AI Agent Engine and Gemini Enterprise."
    )
    parser.add_argument("--region", default=settings.agent_engine_region)
    parser.add_argument("--display-name", default=DEFAULT_DISPLAY_NAME)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument(
        "--service-account",
        default=os.getenv("AGENT_SERVICE_ACCOUNT"),
        help="Runtime service account email for Agent Engine.",
    )
    parser.add_argument(
        "--skip-ge-register",
        action="store_true",
        help="Deploy to Vertex AI Agent Engine only without registering in Gemini Enterprise.",
    )
    args = parser.parse_args()

    if is_placeholder_value(settings.project_id) or is_placeholder_value(
        settings.project_number
    ):
        print(
            "ERROR: Configure GCP_PROJECT_ID and GCP_PROJECT_NUMBER in .env or environment "
            "variables before deploying.",
            file=sys.stderr,
        )
        return 1

    re_name = deploy_reasoning_engine(
        project_id=settings.project_id,
        project_number=settings.project_number,
        region=args.region,
        display_name=args.display_name,
        description=args.description,
        service_account=args.service_account,
    )

    if not args.skip_ge_register:
        if is_placeholder_value(settings.gemini_enterprise_engine_id):
            print(
                "Skipping Gemini Enterprise registration: GEMINI_ENTERPRISE_ENGINE_ID is not set."
            )
        else:
            ge_agent = register_in_gemini_enterprise(
                project_id=settings.project_id,
                project_number=settings.project_number,
                ge_engine_id=settings.gemini_enterprise_engine_id,
                reasoning_engine_name=re_name,
                display_name=args.display_name,
                description=args.description,
            )
            print(json.dumps(ge_agent, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
