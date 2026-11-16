# activeFrontier

A Python package for simulating spatio-temporal species dispersal dynamics and biotic interactions (sympatry, competition, exclusion, obligate, facultative, parasitism, and amensalism) across geographic landscapes, using raster suitability surfaces and occurrence records.

## Prerequisites

- **Python** >= 3.9
- **FFmpeg**, installed and accessible in your system `PATH` (required only if `generate_gif=True`).

## Installation

You can install the package directly from GitHub via `pip`:

```bash
pip install git+https://github.com/PatronRiveroC/activeFrontier.git
```

## Usage

Below is a minimal example. Replace the function and parameter names with the ones defined in your package's API.

```python
from activefrontier import run_simulation

run_simulation(
    suitability_raster="data/suitability.tif",   # raster suitability surface
    occurrences="data/occurrences.csv",          # occurrence records
    steps=100,                                   # number of simulation steps
    interaction="competition",                   # biotic interaction type
    output_dir="results/",                       # where outputs are saved
    generate_gif=True,                           # requires FFmpeg
)
```

Supported biotic interactions: `sympatry`, `competition`, `exclusion`, `obligate`, `facultative`, `parasitism`, and `amensalism`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
