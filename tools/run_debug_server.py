"""Run the CompassCart debug shell with the Python standard library server."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from socketserver import ThreadingMixIn
from typing import Any
from wsgiref.simple_server import WSGIServer, make_server

from compasscart_debug.config import DebugConfig
from compasscart_debug.http import create_application


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CompassCart debug server")
    parser.parse_args(argv)
    config = DebugConfig.from_env()
    app = create_application(config)
    server: Any | None = None
    try:
        server = make_server(
            config.host,
            config.port,
            app,
            server_class=ThreadingWSGIServer,
        )
        print(f"http://{config.host}:{config.port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        if server is not None:
            server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
