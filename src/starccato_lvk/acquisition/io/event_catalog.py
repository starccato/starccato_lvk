import os

import pandas as pd
from pycbc.catalog import Catalog

HERE = os.path.dirname(os.path.abspath(__file__))
EVENT_CSV = f"{HERE}/event_catalog.csv"


def load_event_catalog():
    """
    Load event catalog from PyCBC.

    Parameters
    ----------
    catalog_name : str
        Name of the catalog to load (e.g., "O3b").
    catalog_type : str
        Type of the catalog to load ("full" or "blip").

    Returns
    -------
    Catalog
        Loaded event catalog.
    """
    if not os.path.exists(EVENT_CSV):
        gwtc2 = Catalog(source="gwtc-2")
        gwtc3 = Catalog(source="gwtc-3")
        events = []
        for c in [gwtc2, gwtc3]:
            for event in c:
                events.append([event['commonName'], event['time']])
        # Save to CSV
        df = pd.DataFrame(events, columns=['event_name', 'event_time'])
        df.to_csv(EVENT_CSV, index=False)

    return pd.read_csv(EVENT_CSV)


EVENT_CATALOG = load_event_catalog()
