"""Provenance helpers shared by the data builders."""
from __future__ import annotations

import os


def data_date() -> str:
    """Retrieval date written into provenance records for the prepared city data.

    Set RIYADHAI_DATA_DATE=YYYY-MM-DD in the environment (or .env) when you download or build the data; otherwise provenance says the date was not recorded
    instead of claiming a date that may be wrong.
    """
    return os.environ.get("RIYADHAI_DATA_DATE", "date not recorded")
