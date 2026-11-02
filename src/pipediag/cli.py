
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser


def _run_dashboard(
    host: str,
    port: int,
    open_browser: bool,
) -> int:
    """Launch the existing PipeDiag FastAPI dashboard."""

    try:
        import uvicorn
    except ImportError:
        print(
            "Dashboard dependencies are missing.\n"
            'Install them using: pip install "pipediag[web]"',
            file=sys.stderr,
        )
        return 1

    url = f"http://{host}:{port}"

    print("Starting PipeDiag dashboard...")
    print(f"Dashboard URL: {url}")
    print("Press CTRL+C to stop the server.")

    if open_browser:
        threading.Timer(
            1.5,
            lambda: webbrowser.open(url),
        ).start()

    uvicorn.run(
        "web.backend.main:app",
        host=host,
        port=port,
        reload=False,
    )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipediag",
        description="PipeDiag command-line interface",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the PipeDiag web dashboard",
    )

    dashboard_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)",
    )

    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (default: 8000)",
    )

    dashboard_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser",
    )

    args = parser.parse_args(argv)

    if args.command == "dashboard":
        if not 1 <= args.port <= 65535:
            parser.error("Port must be between 1 and 65535.")

        return _run_dashboard(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
