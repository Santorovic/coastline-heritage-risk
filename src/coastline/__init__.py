"""COASTLINE: multi-hazard risk framework for coastal archaeological heritage.

Hazard (H), Vulnerability (V) and Exposure (E) indices combined into a Risk
Index (R) for 34 archaeological sites in Cyprus, from open geospatial data.
"""
import logging

__version__ = "1.0.0"


def setup_logging(level: int = logging.INFO) -> None:
    """Readable log messages for the scripts: time, module, message."""
    logging.basicConfig(level=level, format="%(asctime)s  %(name)-28s  %(message)s", datefmt="%H:%M:%S")
