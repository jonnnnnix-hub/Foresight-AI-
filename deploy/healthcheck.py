#!/usr/bin/env python3
"""Minimal HTTP health check server for Fly.io.

Runs on port 9090, returns 200 with bot status JSON.
Checks supervisord for running processes.
"""

import http.server
import json
import subprocess
import threading


def get_bot_status() -> dict:
    """Query supervisord for bot process status."""
    try:
        result = subprocess.run(
            ["supervisorctl", "-c", "/etc/supervisor/conf.d/flowedge.conf", "status"],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        bots = {}
        running = 0
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0].replace("bots:", "")
                status = parts[1]
                bots[name] = status
                if status == "RUNNING":
                    running += 1
        return {"ok": running > 0, "running": running, "total": len(bots), "bots": bots}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            status = get_bot_status()
            code = 200 if status.get("ok") else 503
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress access logs


def start_health_server():
    server = http.server.HTTPServer(("0.0.0.0", 9090), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("[healthcheck] listening on :9090/health")


if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", 9090), HealthHandler)
    print("[healthcheck] listening on :9090/health")
    server.serve_forever()
