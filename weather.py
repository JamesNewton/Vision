#!/usr/bin/env python3
import json
import os
import re
import socket
import struct
import requests

import logging
import os

level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, level_name, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

STATE_FILE = "config.json"
USER_AGENT = "(nws-mqtt-alerts, your-email@example.com)"  # Required by NWS

MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
MQTT_USER = None  # Set to string if your broker requires auth
MQTT_PASS = None


# --- Raw MQTT 3.1.1 Publisher (QoS 0, no external library) ---

def _encode_len(length: int) -> bytes:
    out = bytearray()
    while True:
        byte = length % 128
        length //= 128
        if length > 0:
            byte |= 0x80
        out.append(byte)
        if length == 0:
            break
    return bytes(out)


def _encode_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("!H", len(raw)) + raw

def mqtt_publish(topic: str, payload: str, retain: bool = False) -> None:
    # 1. Build CONNECT packet
    flags = 0x02  # Clean session
    body = _encode_str("nws-alert-pub")
    if MQTT_USER:
        flags |= 0x80
        body += _encode_str(MQTT_USER)
        if MQTT_PASS:
            flags |= 0x40
            body += _encode_str(MQTT_PASS)

    var_header = _encode_str("MQTT") + bytes([0x04, flags]) + struct.pack("!H", 60)
    connect_pkt = b"\x10" + _encode_len(len(var_header) + len(body)) + var_header + body

    # 2. Build PUBLISH packet (QoS 0)
    pub_header = bytes([0x31 if retain else 0x30])
    pub_body = _encode_str(topic) + payload.encode("utf-8")
    publish_pkt = pub_header + _encode_len(len(pub_body)) + pub_body

    # 3. Send over raw TCP socket
    with socket.create_connection((MQTT_HOST, MQTT_PORT), timeout=5) as sock:
        sock.sendall(connect_pkt)
        connack = sock.recv(4)
        if len(connack) < 4 or connack[3] != 0:
            raise RuntimeError(f"MQTT broker rejected connection: {connack!r}")
        sock.sendall(publish_pkt)
        sock.sendall(b"\xe0\x00")  # DISCONNECT


# --- Geocoding & NWS Alert Logic ---

def geocode_address(address: str):
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "format": "json",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    matches = resp.json().get("result", {}).get("addressMatches", [])
    if not matches:
        raise ValueError(f"Could not geocode address: {address}")
    coords = matches[0]["coordinates"]
    return round(coords["y"], 4), round(coords["x"], 4)  # lat, lon


def slugify(event_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", event_name.lower()).strip("_")


def main() -> None:
    if not os.path.exists(STATE_FILE):
        raise FileNotFoundError(f"Create {STATE_FILE} with an 'address' or 'lat'/'lon' first.")

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    # Step 1: Resolve lat/lon once if missing
    if state.get("lat") is None or state.get("lon") is None:
        address = state.get("address")
        if not address:
            raise ValueError("Neither lat/lon nor address is set in config.")
        state["lat"], state["lon"] = geocode_address(address)
        logging.log("lat/lon found:", state["lat"],state["lon"])

    # Step 2: Query NWS active alerts for the point
    url = f"https://api.weather.gov/alerts/active?point={state['lat']},{state['lon']}"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except (requests.RequestException, ValueError) as exc:
        err_msg = f"NWS request failed: {exc}"
        logging.error(err_msg)
        try:
            mqtt_publish("weather/error", json.dumps({"error": err_msg}))
        except OSError as mqtt_exc:
            logging.error("Failed to publish error to MQTT: %s", mqtt_exc)
        return
    logging.debug("Recieved: %s", features)

    # Step 3: Publish unseen alerts to MQTT
    seen = set(state.get("seen_ids", []))
    active_ids = set()

    for feature in features:
        props = feature.get("properties", {})
        alert_id = props.get("id")
        if not alert_id:
            continue
        active_ids.add(alert_id)

        if alert_id not in seen:
            event = props.get("event", "unknown")
            topic = f"weather/{slugify(event)}"
            payload = json.dumps({
                "event": event,
                "severity": props.get("severity"),
                "urgency": props.get("urgency"),
                "headline": props.get("headline"),
                "expires": props.get("expires"),
            })
            mqtt_publish(topic, payload)
            logging.log("pub %s %s", topic, payload)

    # Prune expired alert IDs so the file stays small
    state["seen_ids"] = list(active_ids)

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


if __name__ == "__main__":
    main()
