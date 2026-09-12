# Stock ML Platform

The project provides a browser dashboard backed by the existing OI/KNN prediction logic. It accepts exactly five manually entered market inputs:

- current_nifty
- call_oi_change
- put_oi_change
- total_call_oi
- total_put_oi

## Run the dashboard

Start the local server:

```bash
python app.py
```

Open `http://localhost:8000/` in a browser. Enter the five values and select **ANALYZE NIFTY**.

The dashboard validates the fields, shows a loading state, sends `POST /api/predict`, and renders the structured analysis without displaying raw console output.

### Open it on a phone or tablet

Run `python app.py` on the computer hosting the project, keep the phone/tablet on the same Wi-Fi, and open the computer's local IPv4 address with port `8000`, for example:

```text
http://192.168.29.154:8000/
```

The interface automatically adapts to phone, tablet, and desktop widths. The API uses relative browser requests, so the same LAN address works for the dashboard and prediction requests.

## Run the prediction flow from the command line

Interactive CLI:

```bash
python predict.py
```

Pass a JSON payload directly:

```bash
python predict.py --json '{"current_nifty":23352,"call_oi_change":2142000,"put_oi_change":587000,"total_call_oi":230200000,"total_put_oi":254200000}'
```

Start the JSON API directly:

```bash
python app.py
```

Then send a POST to `/api/predict` with the same payload. The root URL serves the dashboard and `/health` returns the server status.

The backend evaluates the existing OI/KNN logic and returns a structured result with the primary signal, KNN pattern, confidence, prediction values, similar candles, PCR analysis, risk data, invalidation conditions, strategy data, and smart-money analysis. Values unavailable from the historical model are rendered as `Not available`; no analytical values are fabricated.
