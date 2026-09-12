import os

from predict import _run_server

if __name__ == "__main__":
    # Render supplies PORT; local runs continue to use port 8000.
    port = int(os.environ.get("PORT", "8000"))
    _run_server(port=port, host="0.0.0.0")
