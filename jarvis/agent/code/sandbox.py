"""
Secure Docker-based sandbox for code execution.
"""

from __future__ import annotations

import os
import docker
import tempfile
import logging

logger = logging.getLogger(__name__)

try:
    client = docker.from_env()
except Exception as e:
    logger.error(f"Failed to connect to Docker: {e}")
    client = None

def execute_code(code: str, language: str = "python", timeout: int = 15) -> dict:
    """
    Execute code in a restricted Docker container.
    Returns: {"success": bool, "output": str, "error": str}
    """
    if client is None:
        return {"success": False, "output": "", "error": "Docker is not available on this system."}

    # Image choice based on language
    image = "python:3.11-slim"
    command = ["python", "-c", code]
    
    if language == "bash":
        image = "debian:stable-slim"
        command = ["bash", "-c", code]

    try:
        # Create a restricted container
        # - Network: none
        # - CPU/Memory limits
        # - Read-only rootfs (optional, might break some things)
        container = client.containers.run(
            image=image,
            command=command,
            network_disabled=True,
            mem_limit="128m",
            cpu_quota=50000,  # 50% of one core
            detach=True,
            stderr=True,
            stdout=True,
        )

        try:
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            exit_code = result.get("StatusCode", 1)
            output = container.logs(stdout=True, stderr=False).decode("utf-8")
            error = container.logs(stdout=False, stderr=True).decode("utf-8")
            
            return {
                "success": exit_code == 0,
                "output": output,
                "error": error
            }
        except Exception as e:
            container.kill()
            return {"success": False, "output": "", "error": f"Execution timed out or failed: {e}"}
        finally:
            container.remove(force=True)

    except Exception as e:
        return {"success": False, "output": "", "error": f"Failed to run container: {e}"}
