"""activeFrontier: cellular-automata simulation of species dispersal and
biotic interactions on active colonization frontiers."""

from importlib.metadata import PackageNotFoundError, version

from .single_sim import single_sim
from .sympatry_sim import sympatry_sim
from .competition_sim import competition_sim
from .exclusion_sim import exclusion_sim
from .obligate_sim import obligate_sim
from .facultative_sim import facultative_sim
from .parasitism_sim import parasitism_sim
from .amensalism_sim import amensalism_sim
from .avoidance_sim import avoidance_sim
from .utils import get_example_data, vignette

try:
    __version__ = version("activefrontier")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.1.0"

__all__ = [
    "single_sim",
    "sympatry_sim",
    "competition_sim",
    "exclusion_sim",
    "obligate_sim",
    "facultative_sim",
    "parasitism_sim",
    "amensalism_sim",
    "avoidance_sim",
    "get_example_data",
    "vignette",
]
