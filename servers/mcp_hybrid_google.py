"""FastMCP entrypoint exposing the FancyRAG hybrid retriever."""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import sys

from dotenv import load_dotenv
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from starlette.responses import JSONResponse

from fancyrag.config import ConfigurationError, load_config
from fancyrag.logging_setup import configure_logging
from fancyrag.mcp.runtime import build_server, create_state


def main() -> int:
    configure_logging()
    load_dotenv(".env.local", override=False)

    logger = logging.getLogger("fancyrag.server")

    try:
        config = load_config()
    except ConfigurationError as error:
        logger.error("server.config_error", extra={"error": str(error)})
        return 1

    try:
        state = create_state(config)
    except Exception as error:  # pragma: no cover - defensive startup guard
        logger.exception("server.startup_failed", extra={"error": type(error).__name__})
        return 1

    static_token = os.getenv("MCP_STATIC_TOKEN")
    auth_provider = None
    if config.server.auth_required and static_token:
        auth_provider = StaticTokenVerifier(
            tokens={
                static_token: {
                    "client_id": "static-token",
                    "scopes": config.oauth.required_scopes,
                }
            },
            required_scopes=config.oauth.required_scopes,
        )
        logger.info(
            "server.static_token_auth_enabled",
            extra={"scopes": config.oauth.required_scopes},
        )

    server = build_server(state, auth_provider=auth_provider)

    @server.custom_route("/mcp/health", methods=["GET"], name="mcp_health")
    async def health_check(_request):
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, state.driver.verify_connectivity)
        except Exception as exc:  # pragma: no cover - defensive health guard
            logger.warning(
                "server.healthcheck_failed", extra={"error": type(exc).__name__}
            )
            return JSONResponse(
                {"status": "unhealthy", "reason": type(exc).__name__}, status_code=503
            )

        return JSONResponse({"status": "ok"})

    @atexit.register
    def _cleanup() -> None:  # pragma: no cover - executed at interpreter shutdown
        state.driver.close()

    logger.info(
        "server.starting",
        extra={
            "host": config.server.host,
            "port": config.server.port,
            "path": config.server.path,
        },
    )

    server.run(
        transport="http",
        host=config.server.host,
        port=config.server.port,
        path=config.server.path,
        stateless_http=True,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
