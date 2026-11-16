"""Helpers to access the bundled example data and the HTML vignette."""

import webbrowser
from importlib import resources
from pathlib import Path


def _package_dir(name):
    return Path(str(resources.files("activefrontier") / name))


def get_example_data(filename=None):
    """Return the path (str) of the bundled example data.

    Without arguments, returns the folder that holds ``rA.tif``, ``rB.tif``,
    ``occA.csv``, ``occB.csv`` and ``hidro.shp``. With ``filename``, returns
    the full path of that file.
    """
    data_dir = _package_dir("data")
    if filename is None:
        return str(data_dir)
    path = data_dir / filename
    if not path.exists():
        available = sorted(p.name for p in data_dir.iterdir())
        raise FileNotFoundError(
            f"'{filename}' is not in the bundled data. Available files: {available}"
        )
    return str(path)


def vignette(open_browser=True):
    """Open the HTML vignette in the default web browser.

    Returns the path of the vignette file, so it can also be opened manually.
    """
    path = _package_dir("vignette") / "vignette.html"
    if not path.exists():
        raise FileNotFoundError(f"Vignette not found at {path}")
    if open_browser:
        webbrowser.open(path.as_uri())
    return str(path)
