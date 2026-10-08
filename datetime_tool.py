"""
Date and Time Tool Implementation (Standard Library Only)
---------------------------------------------------------
Retrieves current date, time, weekday, and timezone information.
Zero external frameworks required.
"""

from datetime import datetime
from typing import Any, Dict, Optional
import zoneinfo


def execute_get_datetime(timezone: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the current date and time information.
    
    Args:
        timezone: Optional string timezone identifier (e.g. 'UTC', 'Asia/Kolkata').
        
    Returns:
        dict: JSON-serializable dictionary with date, time, day, year, and timezone.
    """
    tz_obj = None
    applied_tz = "Local"

    if timezone and timezone.strip():
        tz_name = timezone.strip()
        try:
            tz_obj = zoneinfo.ZoneInfo(tz_name)
            applied_tz = tz_name
        except Exception:
            # If invalid timezone passed, fallback to UTC or Local gracefully
            try:
                # Try common aliases if passed
                aliases = {
                    "est": "America/New_York",
                    "pst": "America/Los_Angeles",
                    "cst": "America/Chicago",
                    "ist": "Asia/Kolkata",
                    "gmt": "UTC"
                }
                alias_match = aliases.get(tz_name.lower())
                if alias_match:
                    tz_obj = zoneinfo.ZoneInfo(alias_match)
                    applied_tz = alias_match
            except Exception:
                pass

    if tz_obj is not None:
        now = datetime.now(tz_obj)
    else:
        # Local system timezone with tzinfo
        now = datetime.now().astimezone()
        applied_tz = str(now.tzname()) if now.tzname() else "Local"

    return {
        "status": "success",
        "current_date": now.strftime("%Y-%m-%d"),
        "current_time": now.strftime("%H:%M:%S"),
        "current_datetime_iso": now.isoformat(),
        "day_of_week": now.strftime("%A"),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "timezone": applied_tz
    }


if __name__ == "__main__":
    import json
    # Self-test
    res_local = execute_get_datetime()
    print("Local Date & Time:")
    print(json.dumps(res_local, indent=2))
    
    res_utc = execute_get_datetime("UTC")
    print("\nUTC Date & Time:")
    print(json.dumps(res_utc, indent=2))
