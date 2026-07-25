# Data Generation Assumptions & Attack Taxonomy

## Generator Settings
- **Total sessions**: 77,896
- **Entities**: 120 (78 user, 30 service_account, 12 edge_device)
- **Date range**: 2026-01-01 to 2026-04-01 (90 days)
- **Random seed**: 42 (fully reproducible)
- **Anomaly injection rate**: ~2.0% true attacks + ~1.5% edge cases

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
| `normal` | 77,160 | 99.06% |
| `brute_force` | 266 | 0.34% |
| `low_and_slow_exfiltration` | 257 | 0.33% |
| `edge_case` | 100 | 0.13% |
| `lateral_movement` | 62 | 0.08% |
| `impossible_travel` | 22 | 0.03% |
| `credential_stuffing` | 20 | 0.03% |
| `device_spoofing` | 9 | 0.01% |