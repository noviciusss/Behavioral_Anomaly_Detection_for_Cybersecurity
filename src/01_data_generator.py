"""
Phase 1 — Synthetic Data Generator
====================================
Generates realistic behavioral log data for 50-200 entities (users,
service accounts, edge devices) with injected attack patterns.

Outputs:
  data/data_with_labels.csv    — full dataset with ground-truth labels
  data/data_for_inference.csv  — labels stripped (simulates production)
  docs/data_assumptions.md     — generator assumptions + attack taxonomy
"""

import random
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ─── Config ──────────────────────────────────────────────────────────────────
N_ENTITIES        = 120          # total entities across all types
N_SESSIONS        = 80_000       # approximate total session rows
ANOMALY_RATE      = 0.02         # 2% of sessions are anomalous (0.5–3% range)
EDGE_CASE_RATE    = 0.015        # insider drift — NOT treated as anomaly
OUTPUT_DIR        = "data"

ENTITY_TYPE_DIST  = {"user": 0.65, "service_account": 0.25, "edge_device": 0.10}

# Resources available in the environment
ALL_RESOURCES = [
    "db-prod-01", "db-staging-02", "file-server-03", "api-gateway-04",
    "admin-console", "hr-portal", "finance-dashboard", "source-code-repo",
    "backup-server", "monitoring-system", "email-server", "vpn-gateway",
    "identity-provider", "analytics-warehouse", "secret-manager"
]

# Command vocabularies per entity type
COMMAND_VOCAB = {
    "user":            ["login", "logout", "read", "write", "search", "download", "upload", "share", "delete", "modify"],
    "service_account": ["api_call", "token_refresh", "health_check", "data_sync", "batch_read", "batch_write", "rotate_key"],
    "edge_device":     ["heartbeat", "telemetry_push", "firmware_check", "config_pull", "alert_push", "sensor_read"],
}

AUTH_METHODS = ["password", "mfa_totp", "mfa_push", "sso_saml", "cert_based", "api_key"]

# ─── Entity Profile Generation ───────────────────────────────────────────────

def generate_geo():
    """Return (lat, lon, city_label) for a plausible corporate location."""
    locations = [
        (40.71, -74.01, "New York"),
        (37.77, -122.42, "San Francisco"),
        (51.51, -0.13,  "London"),
        (48.85,  2.35,  "Paris"),
        (35.69, 139.69, "Tokyo"),
        (1.35,  103.82, "Singapore"),
        (19.08,  72.88, "Mumbai"),
        (-33.87, 151.21, "Sydney"),
    ]
    return random.choice(locations)

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def build_entity_profile(entity_id, entity_type):
    """Create a stable 'home' behavioural profile for one entity."""
    home_lat, home_lon, home_city = generate_geo()
    home_resources = random.sample(ALL_RESOURCES, k=random.randint(2, 5))
    home_device    = fake.uuid4()
    home_hours     = sorted(random.sample(range(6, 22), k=random.randint(4, 8)))  # typical work hours
    home_auth      = random.choice(AUTH_METHODS)
    normal_session_dur = random.uniform(120, 3600)      # seconds
    normal_cmd_len     = random.randint(3, 15)

    return {
        "entity_id":        entity_id,
        "entity_type":      entity_type,
        "home_lat":         home_lat,
        "home_lon":         home_lon,
        "home_city":        home_city,
        "home_resources":   home_resources,
        "home_device":      home_device,
        "home_hours":       home_hours,
        "home_auth":        home_auth,
        "normal_session_dur": normal_session_dur,
        "normal_cmd_len":   normal_cmd_len,
    }

def build_entity_roster(n=N_ENTITIES):
    """Return a list of entity profile dicts."""
    roster = []
    counts = {etype: max(1, int(n * frac)) for etype, frac in ENTITY_TYPE_DIST.items()}
    idx = 0
    for etype, count in counts.items():
        prefix = {"user": "USR", "service_account": "SVC", "edge_device": "DEV"}[etype]
        for i in range(count):
            entity_id = f"{prefix}_{idx:04d}"
            roster.append(build_entity_profile(entity_id, etype))
            idx += 1
    return roster

# ─── Normal Session Generator ────────────────────────────────────────────────

def generate_normal_session(profile, base_time):
    """Generate one normal session row for an entity."""
    # Add noise to location (occasional travel is normal)
    jitter = 0.05  # 5% chance of a nearby-city hop — normal business travel
    if random.random() < jitter:
        lat_noise = np.random.normal(0, 3)
        lon_noise = np.random.normal(0, 3)
        lat = profile["home_lat"] + lat_noise
        lon = profile["home_lon"] + lon_noise
        city = f"Near {profile['home_city']}"
    else:
        lat, lon, city = profile["home_lat"], profile["home_lon"], profile["home_city"]

    hour = random.choice(profile["home_hours"])
    ts   = base_time.replace(hour=hour, minute=random.randint(0, 59),
                              second=random.randint(0, 59))

    vocab    = COMMAND_VOCAB[profile["entity_type"]]
    cmd_len  = max(1, int(np.random.normal(profile["normal_cmd_len"], 2)))
    commands = [random.choice(vocab) for _ in range(cmd_len)]

    # Very occasional new resource (normal curiosity)
    if random.random() < 0.05:
        resource = random.choice(ALL_RESOURCES)
    else:
        resource = random.choice(profile["home_resources"])

    # Rare device switch (new laptop, etc.)
    device = profile["home_device"] if random.random() > 0.03 else fake.uuid4()

    # Mostly success
    failed_auth = random.random() < 0.04

    session_dur = max(10, np.random.normal(profile["normal_session_dur"],
                                            profile["normal_session_dur"] * 0.2))
    return {
        "entity_id":         profile["entity_id"],
        "entity_type":       profile["entity_type"],
        "timestamp":         ts.isoformat(),
        "source_ip":         fake.ipv4_private(),
        "geo_lat":           round(lat, 4),
        "geo_lon":           round(lon, 4),
        "geo_location":      city,
        "resource_accessed": resource,
        "auth_method":       profile["home_auth"],
        "auth_success":      not failed_auth,
        "session_duration":  round(session_dur, 2),
        "command_sequence":  "|".join(commands),
        "device_fingerprint": device,
        "label":             "normal",
    }

# ─── Attack Injection Functions ──────────────────────────────────────────────

def inject_brute_force(profile, base_time):
    """Burst of failed auth attempts from one IP in a short window."""
    records = []
    attack_ip = fake.ipv4_public()
    for i in range(random.randint(15, 40)):
        ts = base_time + timedelta(seconds=i * random.randint(1, 10))
        records.append({
            "entity_id":          profile["entity_id"],
            "entity_type":        profile["entity_type"],
            "timestamp":          ts.isoformat(),
            "source_ip":          attack_ip,
            "geo_lat":            profile["home_lat"],
            "geo_lon":            profile["home_lon"],
            "geo_location":       profile["home_city"],
            "resource_accessed":  random.choice(["admin-console", "identity-provider"]),
            "auth_method":        "password",
            "auth_success":       i == random.randint(30, 40),  # one success at the end
            "session_duration":   round(random.uniform(1, 15), 2),
            "command_sequence":   "login",
            "device_fingerprint": fake.uuid4(),
            "label":              "brute_force",
        })
    return records


def inject_impossible_travel(profile, base_time):
    """Two sessions from geos that are physically impossible within the time gap."""
    geo1 = generate_geo()
    geo2 = generate_geo()
    while haversine_km(geo1[0], geo1[1], geo2[0], geo2[1]) < 5000:
        geo2 = generate_geo()

    gap_seconds = random.randint(300, 3600)   # only 5 min–1 h apart
    ts1 = base_time
    ts2 = base_time + timedelta(seconds=gap_seconds)

    return [
        {
            "entity_id":          profile["entity_id"],
            "entity_type":        profile["entity_type"],
            "timestamp":          ts1.isoformat(),
            "source_ip":          fake.ipv4_public(),
            "geo_lat":            geo1[0],
            "geo_lon":            geo1[1],
            "geo_location":       geo1[2],
            "resource_accessed":  random.choice(profile["home_resources"]),
            "auth_method":        profile["home_auth"],
            "auth_success":       True,
            "session_duration":   round(random.uniform(60, 600), 2),
            "command_sequence":   "|".join([random.choice(COMMAND_VOCAB[profile["entity_type"]]) for _ in range(5)]),
            "device_fingerprint": profile["home_device"],
            "label":              "impossible_travel",
        },
        {
            "entity_id":          profile["entity_id"],
            "entity_type":        profile["entity_type"],
            "timestamp":          ts2.isoformat(),
            "source_ip":          fake.ipv4_public(),
            "geo_lat":            geo2[0],
            "geo_lon":            geo2[1],
            "geo_location":       geo2[2],
            "resource_accessed":  random.choice(profile["home_resources"]),
            "auth_method":        profile["home_auth"],
            "auth_success":       True,
            "session_duration":   round(random.uniform(60, 600), 2),
            "command_sequence":   "|".join([random.choice(COMMAND_VOCAB[profile["entity_type"]]) for _ in range(5)]),
            "device_fingerprint": profile["home_device"],
            "label":              "impossible_travel",
        }
    ]


def inject_credential_stuffing(profiles, base_time, n_victims=20):
    """Many entities, few source IPs, high failure rate — population-level signal."""
    records = []
    attack_ips = [fake.ipv4_public() for _ in range(random.randint(2, 5))]
    victims = random.sample(profiles, min(n_victims, len(profiles)))
    for profile in victims:
        ts = base_time + timedelta(seconds=random.randint(0, 600))
        records.append({
            "entity_id":          profile["entity_id"],
            "entity_type":        profile["entity_type"],
            "timestamp":          ts.isoformat(),
            "source_ip":          random.choice(attack_ips),
            "geo_lat":            profile["home_lat"],
            "geo_lon":            profile["home_lon"],
            "geo_location":       profile["home_city"],
            "resource_accessed":  "identity-provider",
            "auth_method":        "password",
            "auth_success":       random.random() < 0.05,
            "session_duration":   round(random.uniform(1, 30), 2),
            "command_sequence":   "login",
            "device_fingerprint": fake.uuid4(),
            "label":              "credential_stuffing",
        })
    return records


def inject_lateral_movement(profile, base_time):
    """Compromised entity rapidly touches many new resources."""
    records = []
    new_resources = random.sample([r for r in ALL_RESOURCES if r not in profile["home_resources"]],
                                   k=random.randint(4, 8))
    for i, resource in enumerate(new_resources):
        ts = base_time + timedelta(minutes=i * random.randint(2, 10))
        vocab    = COMMAND_VOCAB[profile["entity_type"]]
        commands = [random.choice(vocab) for _ in range(random.randint(3, 8))]
        records.append({
            "entity_id":          profile["entity_id"],
            "entity_type":        profile["entity_type"],
            "timestamp":          ts.isoformat(),
            "source_ip":          fake.ipv4_private(),
            "geo_lat":            profile["home_lat"],
            "geo_lon":            profile["home_lon"],
            "geo_location":       profile["home_city"],
            "resource_accessed":  resource,
            "auth_method":        profile["home_auth"],
            "auth_success":       True,
            "session_duration":   round(random.uniform(30, 300), 2),
            "command_sequence":   "|".join(commands),
            "device_fingerprint": profile["home_device"],
            "label":              "lateral_movement",
        })
    return records


def inject_device_spoofing(profile, base_time):
    """Same device_id as a known device, but mismatched fingerprint."""
    spoofed_device = profile["home_device"]   # same ID, different fingerprint characteristics
    # We mark this by appending "_SPOOFED" to the fingerprint value — the detector
    # compares fingerprint hash, not just the ID.
    spoofed_fp = spoofed_device + "_SPOOFED_" + fake.md5()[:8]
    vocab    = COMMAND_VOCAB[profile["entity_type"]]
    commands = [random.choice(vocab) for _ in range(random.randint(5, 15))]
    return [{
        "entity_id":          profile["entity_id"],
        "entity_type":        profile["entity_type"],
        "timestamp":          base_time.isoformat(),
        "source_ip":          fake.ipv4_public(),
        "geo_lat":            profile["home_lat"],
        "geo_lon":            profile["home_lon"],
        "geo_location":       profile["home_city"],
        "resource_accessed":  random.choice(profile["home_resources"]),
        "auth_method":        profile["home_auth"],
        "auth_success":       True,
        "session_duration":   round(random.uniform(60, 1800), 2),
        "command_sequence":   "|".join(commands),
        "device_fingerprint": spoofed_fp,
        "label":              "device_spoofing",
    }]


def inject_low_and_slow(profile, base_time, n_days=30):
    """Gradual off-hours access growth spread over many days — the hardest to catch."""
    records = []
    exfil_resource = random.choice(["backup-server", "analytics-warehouse", "finance-dashboard"])
    for day in range(n_days):
        # Only active 40% of nights to keep it subtle
        if random.random() > 0.4:
            continue
        # 1–3 sessions per night, growing slightly in volume each week
        n_sessions = 1 + (day // 10)
        for _ in range(n_sessions):
            night_hour = random.randint(0, 5)
            ts = base_time + timedelta(days=day,
                                        hours=night_hour,
                                        minutes=random.randint(0, 59))
            # Gradually increasing download-like commands
            commands = ["read", "download"] * (1 + day // 15)
            records.append({
                "entity_id":          profile["entity_id"],
                "entity_type":        profile["entity_type"],
                "timestamp":          ts.isoformat(),
                "source_ip":          fake.ipv4_private(),
                "geo_lat":            profile["home_lat"],
                "geo_lon":            profile["home_lon"],
                "geo_location":       profile["home_city"],
                "resource_accessed":  exfil_resource,
                "auth_method":        profile["home_auth"],
                "auth_success":       True,
                "session_duration":   round(random.uniform(300, 1800), 2),
                "command_sequence":   "|".join(commands),
                "device_fingerprint": profile["home_device"],
                "label":              "low_and_slow_exfiltration",
            })
    return records


def inject_insider_drift(profile, base_time, n_days=60):
    """
    Slow, legitimate-looking privilege creep over 2 months.
    Labelled 'edge_case' — used to tune FP threshold, NOT treated as an attack.
    """
    records = []
    # Gradually access new resources one by one over time
    new_resources = [r for r in ALL_RESOURCES if r not in profile["home_resources"]]
    random.shuffle(new_resources)
    for i, resource in enumerate(new_resources[:5]):
        day_offset = int(i * (n_days / 5)) + random.randint(0, 5)
        ts = base_time + timedelta(days=day_offset,
                                    hours=random.choice(profile["home_hours"]))
        vocab    = COMMAND_VOCAB[profile["entity_type"]]
        commands = [random.choice(vocab) for _ in range(random.randint(2, 6))]
        records.append({
            "entity_id":          profile["entity_id"],
            "entity_type":        profile["entity_type"],
            "timestamp":          ts.isoformat(),
            "source_ip":          fake.ipv4_private(),
            "geo_lat":            profile["home_lat"],
            "geo_lon":            profile["home_lon"],
            "geo_location":       profile["home_city"],
            "resource_accessed":  resource,
            "auth_method":        profile["home_auth"],
            "auth_success":       True,
            "session_duration":   round(random.uniform(120, 1800), 2),
            "command_sequence":   "|".join(commands),
            "device_fingerprint": profile["home_device"],
            "label":              "edge_case",     # NOT "anomaly"
        })
    return records

# ─── Main Generation Pipeline ────────────────────────────────────────────────

def generate_dataset():
    print("=" * 60)
    print("  Synthetic UEBA Data Generator — Phase 1")
    print("=" * 60)

    roster = build_entity_roster()
    print(f"[+] Built {len(roster)} entity profiles")

    all_records = []
    # Date range: ~90 days, starting Jan 1 2026
    start_date = datetime(2026, 1, 1, 8, 0, 0)

    # ── Normal sessions ──────────────────────────────────────────────
    target_normal = int(N_SESSIONS * (1 - ANOMALY_RATE - EDGE_CASE_RATE))
    sessions_per_entity = target_normal // len(roster)

    print(f"[+] Generating ~{target_normal:,} normal sessions...")
    for profile in roster:
        for i in range(sessions_per_entity):
            day_offset  = random.randint(0, 89)
            base_time   = start_date + timedelta(days=day_offset)
            all_records.append(generate_normal_session(profile, base_time))

    # ── Attack injections ────────────────────────────────────────────
    print(f"[+] Injecting attacks (target rate ~{ANOMALY_RATE*100:.1f}%)...")

    anomaly_target = int(N_SESSIONS * ANOMALY_RATE)
    injected = 0
    attack_profiles = random.sample(roster, k=min(50, len(roster)))

    for profile in attack_profiles:
        if injected >= anomaly_target:
            break
        base_time = start_date + timedelta(days=random.randint(0, 89))
        attack = random.choice([
            "brute_force", "impossible_travel", "lateral_movement",
            "device_spoofing", "low_and_slow"
        ])
        if attack == "brute_force":
            records = inject_brute_force(profile, base_time)
        elif attack == "impossible_travel":
            records = inject_impossible_travel(profile, base_time)
        elif attack == "lateral_movement":
            records = inject_lateral_movement(profile, base_time)
        elif attack == "device_spoofing":
            records = inject_device_spoofing(profile, base_time)
        else:
            records = inject_low_and_slow(profile, base_time)
        all_records.extend(records)
        injected += len(records)

    # Credential stuffing — population-level, inject once
    cs_time = start_date + timedelta(days=random.randint(10, 80))
    all_records.extend(inject_credential_stuffing(roster, cs_time))
    print(f"[+] Credential stuffing event injected")

    # ── Insider drift (edge_case) ────────────────────────────────────
    drift_target  = int(N_SESSIONS * EDGE_CASE_RATE)
    drift_profiles = random.sample(roster, k=min(20, len(roster)))
    for profile in drift_profiles:
        base_time = start_date + timedelta(days=random.randint(0, 29))
        all_records.extend(inject_insider_drift(profile, base_time))

    # ── Assemble DataFrame ───────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["session_id"] = [f"SES_{i:07d}" for i in range(len(df))]

    # Reorder columns
    cols = ["session_id", "entity_id", "entity_type", "timestamp",
            "source_ip", "geo_lat", "geo_lon", "geo_location",
            "resource_accessed", "auth_method", "auth_success",
            "session_duration", "command_sequence", "device_fingerprint", "label"]
    df = df[cols]

    # ── Label distribution ───────────────────────────────────────────
    print("\n[+] Label distribution:")
    print(df["label"].value_counts().to_string())
    n_anomaly   = (df["label"] != "normal").sum()
    n_edge_case = (df["label"] == "edge_case").sum()
    true_anomaly = n_anomaly - n_edge_case
    print(f"\n    True anomalies: {true_anomaly:,}  ({true_anomaly/len(df)*100:.2f}%)")
    print(f"    Edge cases:     {n_edge_case:,}  ({n_edge_case/len(df)*100:.2f}%)")
    print(f"    Total rows:     {len(df):,}")

    # ── Save ─────────────────────────────────────────────────────────
    labeled_path   = f"{OUTPUT_DIR}/data_with_labels.csv"
    inference_path = f"{OUTPUT_DIR}/data_for_inference.csv"

    df.to_csv(labeled_path, index=False)
    df.drop(columns=["label"]).to_csv(inference_path, index=False)

    print(f"\n[✓] Saved: {labeled_path}  ({len(df):,} rows)")
    print(f"[✓] Saved: {inference_path}  (labels stripped)")

    write_assumptions_doc(roster, df)
    return df, roster


def write_assumptions_doc(roster, df):
    """Write the data assumptions markdown doc (paste into report later)."""
    label_counts = df["label"].value_counts().to_dict()
    entity_counts = pd.Series([p["entity_type"] for p in roster]).value_counts().to_dict()

    doc = f"""# Data Generation Assumptions & Attack Taxonomy

## Generator Settings
- **Total sessions**: {len(df):,}
- **Entities**: {len(roster)} ({', '.join(f'{v} {k}' for k, v in entity_counts.items())})
- **Date range**: 2026-01-01 to 2026-04-01 (90 days)
- **Random seed**: {SEED} (fully reproducible)
- **Anomaly injection rate**: ~{ANOMALY_RATE*100:.1f}% true attacks + ~{EDGE_CASE_RATE*100:.1f}% edge cases

## Schema
| Field | Type | Description |
|---|---|---|
| `session_id` | string | Unique session identifier |
| `entity_id` | string | Entity identifier (USR_*, SVC_*, DEV_*) |
| `entity_type` | string | user / service_account / edge_device |
| `timestamp` | datetime | Session start time (ISO 8601) |
| `source_ip` | string | Originating IP address |
| `geo_lat`, `geo_lon` | float | Geolocation coordinates |
| `geo_location` | string | City label |
| `resource_accessed` | string | Target resource name |
| `auth_method` | string | Authentication method used |
| `auth_success` | bool | Whether authentication succeeded |
| `session_duration` | float | Session length in seconds |
| `command_sequence` | string | Pipe-delimited ordered command list |
| `device_fingerprint` | string | Device identifier (UUID or spoofed UUID) |
| `label` | string | Ground truth (present only in labeled dataset) |

## Normal Profile Design
Each entity has a stable "home" profile:
- Typical login hours (4–8 hours per day, sampled from 6 AM–10 PM)
- Home geolocation (one of 8 global corporate cities)
- Usual resource set (2–5 resources)
- Preferred auth method (fixed per entity)
- Home device fingerprint (UUID, ~3% chance of switch per session — legitimate device changes)

**Noise injection**: Gaussian/Poisson noise added to session duration, command length, 
and 5% geolocation jitter (short business travel) to prevent trivially clean normal profiles.

## Attack Taxonomy

| Attack | Label | Injection Logic | Key Detection Signal |
|---|---|---|---|
| Brute Force | `brute_force` | 15–40 failed auths from 1 IP in <10 min | Failed-auth rate spike |
| Impossible Travel | `impossible_travel` | Same entity, 2 geos >5000 km apart, <1h gap | Geo-velocity > 900 km/h |
| Credential Stuffing | `credential_stuffing` | 20 entities, 2–5 IPs, 95% failure rate | Population-level signal |
| Lateral Movement | `lateral_movement` | 4–8 new resources in rapid succession | Resource-diversity spike |
| Device Spoofing | `device_spoofing` | Known device_id, mismatched fingerprint hash | Fingerprint inconsistency |
| Low-and-Slow Exfil | `low_and_slow_exfiltration` | Gradual off-hours access growth over 30 days | Long-window trend drift |
| **Insider Drift** | **`edge_case`** | Slow legitimate privilege creep over 60 days | **NOT treated as attack** |

## Label Distribution
| Label | Count | Rate |
|---|---|---|
""" + "\n".join(f"| `{k}` | {v:,} | {v/len(df)*100:.2f}% |"
                for k, v in sorted(label_counts.items(), key=lambda x: -x[1]))

    with open("docs/data_assumptions.md", "w", encoding="utf-8") as f:
        f.write(doc)
    print("[✓] Saved: docs/data_assumptions.md")


if __name__ == "__main__":
    df, roster = generate_dataset()
    print("\n[✓] Phase 1 complete.")
