import argparse
import json
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

TRAIN_PATH = os.path.join(os.path.dirname(__file__), "train.csv")
VALIDATION_MESSAGES = {
    "current_nifty": "Please enter a valid NIFTY price.",
    "call_oi_change": "Please enter a valid call OI change.",
    "put_oi_change": "Please enter a valid put OI change.",
    "total_call_oi": "Please enter a valid total call OI.",
    "total_put_oi": "Please enter a valid total put OI.",
}


def _coerce_float(value):
    if value is None:
        raise ValueError("missing value")
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("empty value")
    cleaned = cleaned.replace(",", "").replace("$", "").replace("+", "")
    cleaned = cleaned.replace("(", "-").replace(")", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        raise ValueError(f"invalid numeric value: {value!r}")
    return float(match.group(0))


def _build_dataframe():
    df = pd.read_csv(TRAIN_PATH)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", na=False)]
    df.columns = [str(c).strip() for c in df.columns]

    if "nifty_before" in df.columns and "nifty_after" in df.columns:
        df["points_change"] = df["nifty_after"] - df["nifty_before"]

    if "TOTAL POINTS UP/DOWN" in df.columns:
        df["TOTAL POINTS UP/DOWN"] = pd.to_numeric(
            df["TOTAL POINTS UP/DOWN"].astype(str).str.replace("+", "", regex=False),
            errors="coerce",
        )

    if "WINNER" in df.columns:
        df["WINNER"] = df["WINNER"].astype(str).str.strip()

    return df


def _calculate_knn(df, live_values):
    feature_names = ["call_volume", "put_volume", "total_call", "total_put"]
    if not all(col in df.columns for col in feature_names):
        raise ValueError("Historical model columns are missing.")

    stds = df[feature_names].std(ddof=0)
    stds = stds.replace(0, 1)
    distances = []
    for _, row in df.iterrows():
        total = 0.0
        for feature, value in zip(feature_names, live_values):
            total += ((row[feature] - value) / stds[feature]) ** 2
        distances.append(total ** 0.5)
    df = df.copy()
    df["distance"] = distances
    near = df.nsmallest(3, "distance").copy()

    call_votes = (near["WINNER"].fillna("").str.strip() == "BUY CALL").sum()
    put_votes = (near["WINNER"].fillna("").str.strip() == "BUY PUT").sum()
    signal = "BUY CALL" if call_votes >= put_votes else "BUY PUT"
    return signal, {"buy_call": int(call_votes), "buy_put": int(put_votes)}, near


def _oi_signal(call_oi_change, put_oi_change):
    if call_oi_change > put_oi_change:
        signal = "BUY PUT"
        reason = (
            f"Call OI increasing more (+{call_oi_change:,.0f} vs +{put_oi_change:,.0f}) "
            "→ Call writers bearish"
        )
    elif put_oi_change > call_oi_change:
        signal = "BUY CALL"
        reason = (
            f"Put OI increasing more (+{put_oi_change:,.0f} vs +{call_oi_change:,.0f}) "
            "→ Put writers bullish"
        )
    else:
        signal = "BUY PUT"
        reason = "Call and Put OI changes are balanced → defaulting to the stronger bearish OI flow"
    return signal, reason


def _determine_risk(near, suggestion):
    if "points_change" not in near.columns or near.empty:
        return {"average_gain": None, "best_case": None, "worst_case": None, "average_loss_when_failed": None}

    wins = near[near["WINNER"].str.strip() == suggestion]["points_change"]
    losses = near[near["WINNER"].str.strip() != suggestion]["points_change"]

    risk = {
        "average_gain": float(wins.mean()) if not wins.empty else None,
        "best_case": float(wins.max()) if not wins.empty else None,
        "worst_case": float(wins.min()) if not wins.empty else None,
        "average_loss_when_failed": float(losses.mean()) if not losses.empty else None,
    }
    return risk


def _build_analysis(payload, near, oi_signal, knn_signal, final_signal):
    feature_names = ["call_volume", "put_volume", "total_call", "total_put"]
    live_values = [
        payload["call_oi_change"],
        payload["put_oi_change"],
        payload["total_call_oi"],
        payload["total_put_oi"],
    ]
    drivers = []
    for feature, live_value in zip(feature_names, live_values):
        if feature in near.columns and not near.empty:
            historical_avg = float(near[feature].mean())
            diff = 0.0 if historical_avg == 0 else ((live_value - historical_avg) / abs(historical_avg)) * 100
            drivers.append(
                {
                    "feature": feature,
                    "live": live_value,
                    "historical_avg": historical_avg,
                    "difference_percent": diff,
                }
            )

    total_call_oi = payload["total_call_oi"]
    total_put_oi = payload["total_put_oi"]
    live_pcr = (total_put_oi / total_call_oi) if total_call_oi else 0.0
    historical_pcr = (
        near["total_put"].mean() / near["total_call"].mean()
        if "total_call" in near.columns and "total_put" in near.columns and not near.empty and near["total_call"].mean() not in (None, 0)
        else 0.0
    )
    if live_pcr > 1.2:
        pcr_interpretation = "Bullish Signal: More Put writing (support building)"
    elif live_pcr < 0.8:
        pcr_interpretation = "Bearish Signal: More Call writing (resistance building)"
    else:
        pcr_interpretation = "Neutral Signal: Balanced OI"

    oi_diff = payload["put_oi_change"] - payload["call_oi_change"]
    if oi_diff > 0:
        oi_flow_summary = {
            "net_oi_flow": oi_diff,
            "market_interpretation": "Put side stronger → support building",
            "primary_signal": "BUY CALL",
        }
    else:
        oi_flow_summary = {
            "net_oi_flow": oi_diff,
            "market_interpretation": "Call side stronger → resistance building",
            "primary_signal": "BUY PUT",
        }

    risk = _determine_risk(near, final_signal)
    expected_change = float(near["points_change"].mean()) if "points_change" in near.columns and not near.empty else 0.0

    if final_signal == "BUY CALL":
        invalidation = [
            f"NIFTY falls below {payload['current_nifty'] - abs(expected_change):.2f}",
            "Put OI stops increasing or decreases",
            "Call OI increases beyond the calculated threshold",
        ]
    else:
        invalidation = [
            f"NIFTY rises above {payload['current_nifty'] + abs(expected_change):.2f}",
            "Call OI stops increasing or decreases",
            "Put OI increases beyond the calculated threshold",
        ]

    strategy = {}
    if "points_change" in near.columns and not near.empty:
        signal_wins = near[near["WINNER"].str.strip() == final_signal]["points_change"]
        avg_gain = float(signal_wins.mean()) if not signal_wins.empty else abs(expected_change)
        strategy = {
            "entry": float(payload["current_nifty"]),
            "target_1": float(payload["current_nifty"] + (avg_gain * 0.5)),
            "target_2": float(payload["current_nifty"] + avg_gain),
            "stop_loss": float(payload["current_nifty"] - abs(avg_gain)),
            "risk_reward": round(abs(avg_gain) / max(abs(avg_gain), 1.0), 2) if avg_gain else 0.0,
        }

    current_pcr = live_pcr
    market_bias = "BULLISH" if current_pcr >= 1.0 else "BEARISH"
    oi_momentum = "BEARISH" if payload["call_oi_change"] > payload["put_oi_change"] else "BULLISH"
    smart_money_reason = "Call writers are adding more than Put writers." if oi_momentum == "BEARISH" else "Put writers are adding more than Call writers."
    smart_money = {
        "current_pcr": round(current_pcr, 3),
        "market_bias": market_bias,
        "oi_momentum": oi_momentum,
        "reason": smart_money_reason,
        "signal": final_signal,
        "signal_divergence": oi_signal != knn_signal,
    }

    return {
        "drivers": drivers,
        "pcr": {
            "live_pcr": round(live_pcr, 3),
            "historical_pcr": round(historical_pcr, 3),
            "interpretation": pcr_interpretation,
        },
        "oi_flow": oi_flow_summary,
        "risk": risk,
        "invalidation": invalidation,
        "strategy": strategy,
        "smart_money": smart_money,
    }


def _prepare_similar_candles(near):
    rows = []
    for _, row in near.iterrows():
        record = {
            "call_oi_change": float(row.get("call_volume", 0.0) if "call_volume" in row.index else 0.0),
            "put_oi_change": float(row.get("put_volume", 0.0) if "put_volume" in row.index else 0.0),
            "total_call_oi": float(row.get("total_call", 0.0) if "total_call" in row.index else 0.0),
            "total_put_oi": float(row.get("total_put", 0.0) if "total_put" in row.index else 0.0),
            "winner": str(row.get("WINNER", "")).strip(),
            "points_change": float(row["points_change"]) if "points_change" in row.index and pd.notna(row.get("points_change")) else None,
        }
        rows.append(record)
    return rows


def build_prediction_response(payload):
    payload = dict(payload)
    required = ["current_nifty", "call_oi_change", "put_oi_change", "total_call_oi", "total_put_oi"]
    missing = [name for name in required if name not in payload or payload[name] in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    parsed = {}
    for key in required:
        try:
            value = _coerce_float(payload[key])
        except ValueError:
            raise ValueError(VALIDATION_MESSAGES[key])
        parsed[key] = float(value)

    if parsed["current_nifty"] <= 0:
        raise ValueError(VALIDATION_MESSAGES["current_nifty"])
    if parsed["total_call_oi"] <= 0:
        raise ValueError(VALIDATION_MESSAGES["total_call_oi"])
    if parsed["total_put_oi"] <= 0:
        raise ValueError(VALIDATION_MESSAGES["total_put_oi"])

    df = _build_dataframe()
    live_values = [
        parsed["call_oi_change"],
        parsed["put_oi_change"],
        parsed["total_call_oi"],
        parsed["total_put_oi"],
    ]
    oi_signal, oi_reason = _oi_signal(parsed["call_oi_change"], parsed["put_oi_change"])
    knn_signal, knn_history, near = _calculate_knn(df, live_values)

    if oi_signal == knn_signal:
        final_signal = oi_signal
        confidence = "HIGH"
        conflict = False
    else:
        final_signal = oi_signal
        confidence = "MEDIUM"
        conflict = True

    expected_change = float(near["points_change"].mean()) if "points_change" in near.columns and not near.empty else 0.0
    target_price = float(parsed["current_nifty"] + expected_change)

    result = {
        "input": {
            "current_nifty": parsed["current_nifty"],
            "call_oi_change": parsed["call_oi_change"],
            "put_oi_change": parsed["put_oi_change"],
            "total_call_oi": parsed["total_call_oi"],
            "total_put_oi": parsed["total_put_oi"],
        },
        "primary_signal": {
            "signal": oi_signal,
            "reason": oi_reason,
        },
        "knn_signal": {
            "signal": knn_signal,
            "history": knn_history,
        },
        "decision": {
            "signal": final_signal,
            "confidence": confidence,
            "conflict": conflict,
            "selected_signal": "OI_FLOW" if conflict else "CONSENSUS",
        },
        "prediction": {
            "expected_change": round(expected_change, 2),
            "target_price": round(target_price, 2),
        },
        "similar_candles": _prepare_similar_candles(near),
        "analysis": _build_analysis(parsed, near, oi_signal, knn_signal, final_signal),
        "metadata": {
            "model_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    return result


class PredictHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/mm.jfif":
            image = (Path(__file__).parent / "mm.jfif").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            self.wfile.write(image)
            return
        if self.path == "/":
            page = (Path(__file__).parent / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path in ("/health", "/api/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "status": "OK",
                        "message": "Your API is running",
                        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    }
                ).encode("utf-8")
            )
            return
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
            result = build_prediction_response(payload)
            status = 200
        except Exception as exc:  # pragma: no cover - server-side guard
            result = {"error": str(exc)}
            status = 400

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, default=str).encode("utf-8"))

    def log_message(self, format, *args):
        return


def _prompt_for_value(label, prompt_text, minimum=None):
    while True:
        try:
            value = float(input(prompt_text))
        except ValueError:
            print(VALIDATION_MESSAGES[label])
            continue
        if minimum is not None and value <= minimum:
            print(VALIDATION_MESSAGES[label])
            continue
        return value


def _interactive_mode():
    current_nifty = _prompt_for_value("current_nifty", "NIFTY Current Price: ", minimum=0)
    call_oi_change = _prompt_for_value("call_oi_change", "Call OI Change: ")
    put_oi_change = _prompt_for_value("put_oi_change", "Put OI Change: ")
    total_call_oi = _prompt_for_value("total_call_oi", "Total Call OI: ", minimum=0)
    total_put_oi = _prompt_for_value("total_put_oi", "Total Put OI: ", minimum=0)
    payload = {
        "current_nifty": current_nifty,
        "call_oi_change": call_oi_change,
        "put_oi_change": put_oi_change,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
    }
    print(json.dumps(build_prediction_response(payload), indent=2))


def _run_server(port=8000, host="127.0.0.1"):
    server = ThreadingHTTPServer((host, port), PredictHandler)
    display_host = "localhost" if host == "127.0.0.1" else "<computer-ip>"
    print(f"Prediction API running on http://{display_host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="NIFTY ML prediction backend")
    parser.add_argument("--json", type=str, help="JSON payload string to analyze")
    parser.add_argument("--server", action="store_true", help="Run HTTP server for /api/predict")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP server")
    args = parser.parse_args()

    if args.server:
        _run_server(port=args.port)
        return

    if args.json:
        try:
            payload = json.loads(args.json)
            print(json.dumps(build_prediction_response(payload), indent=2))
        except Exception as exc:
            print(str(exc))
        return

    _interactive_mode()


if __name__ == "__main__":
    main()
