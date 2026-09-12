from predict import _run_server

if __name__ == "__main__":
    # Bind to all interfaces so phones on the same Wi-Fi can open the dashboard.
    _run_server(host="0.0.0.0")
