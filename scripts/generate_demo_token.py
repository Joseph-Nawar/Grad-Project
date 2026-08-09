"""Generate a signed local Stage 2 demo token."""

from __future__ import annotations

import argparse

from rural_stroke_assist.server.config import ServerSettings
from rural_stroke_assist.server.infrastructure.auth.local_jwt import create_demo_token


def main() -> int:
    settings = ServerSettings.from_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="demo-collector")
    parser.add_argument("--role", action="append", dest="roles", default=["collector"])
    parser.add_argument(
        "--facility", action="append", dest="facilities", default=["demo-facility"]
    )
    args = parser.parse_args()
    print(
        create_demo_token(
            secret=settings.jwt_secret,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            subject=args.subject,
            roles=args.roles,
            facilities=args.facilities,
            ttl_seconds=settings.token_ttl_seconds,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
