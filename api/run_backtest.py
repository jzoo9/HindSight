"""
Vercel serverless function: POST /api/run_backtest
"""
import json
from http.server import BaseHTTPRequestHandler
from main import run_backtest_result


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
            data = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            self._send(400, {"detail": "Invalid JSON body"})
            return

        ticker = (data.get("ticker") or "").strip()
        start_date = data.get("start_date") or ""
        end_date = data.get("end_date") or ""

        if not ticker:
            self._send(400, {"detail": "Ticker is required"})
            return
        if not start_date or not end_date:
            self._send(400, {"detail": "start_date and end_date are required"})
            return

        try:
            result = run_backtest_result(ticker, start_date, end_date)
            self._send(200, result)
        except ValueError as e:
            self._send(400, {"detail": str(e)})
        except Exception as e:
            self._send(500, {"detail": f"Backtest failed: {str(e)}"})

    def _send(self, status: int, body: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def do_GET(self):
        self._send(405, {"detail": "Method not allowed"})

    def log_message(self, format, *args):
        pass
