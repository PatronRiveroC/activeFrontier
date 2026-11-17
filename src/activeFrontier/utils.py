import os
import shutil
import tempfile
import urllib.error
import urllib.request
import webbrowser
from importlib import resources
from pathlib import Path

_REPO = "PatronRiveroC/activeFrontier"
_BRANCH = "main"
_FOLDER = "src/data"

_RAW_URL = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}/{_FOLDER}/"
_BARRIER_FILES = ["hidro.shp", "hidro.shx", "hidro.dbf", "hidro.prj"]
_ALL_SPECIES = ("A", "B", "C", "D", "E", "F", "G", "I", "J")


def _default_cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    return Path(base) / "activefrontier" / "data"


def _download(name, target):
    url = _RAW_URL + urllib.request.quote(name)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(tmp_fd, "wb") as tmp, urllib.request.urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, tmp)
        os.replace(tmp_name, target)
    except Exception as err:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise ConnectionError(f"Could not download '{name}' from {url}. Details: {err}") from err


def get_example_data(species=("A", "B"), dest=None, overwrite=False):

    folder = Path(dest) if dest is not None else _default_cache_dir()
    folder.mkdir(parents=True, exist_ok=True)

    if isinstance(species, str) and species.lower() == "all":
        letters = list(_ALL_SPECIES)
    else:
        letters = [species] if isinstance(species, str) else list(species)

    names = list(_BARRIER_FILES)
    for letter in letters:
        names += [f"r{letter}.tif", f"occ{letter}.csv"]

    for name in names:
        target = folder / name
        if overwrite or not target.exists():
            print(f"Downloading {name}...")
            _download(name, target)

    return str(folder)


def vignette(open_browser=True):

    path = Path(str(resources.files("activefrontier") / "vignette")) / "vignette.html"
    if not path.exists():
        raise FileNotFoundError(f"Vignette not found at {path}")
    if open_browser:
        webbrowser.open(path.as_uri())
    return str(path)
