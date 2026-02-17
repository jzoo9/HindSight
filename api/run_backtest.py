import json
from http.server import BaseHTTPRequestHandler

from api.main import run_backtest_result


def _send_json(h: BaseHTTPRequestHandler, status: int, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")
    h.end_headers()
    h.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(raw.decode("utf-8") or "{}")

            ticker = (req.get("ticker") or "").strip()
            start_date = (req.get("start_date") or "").strip()
            end_date = (req.get("end_date") or "").strip()

            if not ticker or not start_date or not end_date:
                return _send_json(
                    self,
                    400,
                    {"detail": "ticker, start_date, and end_date are required"},
                )

            result = run_backtest_result(ticker, start_date, end_date)
            return _send_json(self, 200, result)
        except Exception as e:
            # Surface the error to the frontend; logs will appear in Vercel too.
            print("api/run_backtest failed:", repr(e))
            return _send_json(self, 500, {"detail": f"Backtest failed: {e}"})
