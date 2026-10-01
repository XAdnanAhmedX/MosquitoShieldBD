from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import networkx as nx
MEDICINES: Dict[str, Dict[str, Any]] = {
    "Permethrin Spray": {
        "targets": ["Aedes Aegypti"],
        "diffusion_radius": 1,      # hops in the graph
        "diffusion_factor": 0.5,    # neighbors get 50% of full coverage
        "units_per_location": 1.0,  # medicine units consumed per sprayed node
        "km_per_unit": 0.6,         # effective spray distance per unit (graph weight units)
        "recommended_when": "Permethrin_Resistant == 0",
        "description": "Standard adulticide targeting adult Aedes Aegypti mosquitoes."
    },
    "Malathion Fog": {
        "targets": ["Aedes Aegypti", "Aedes Albopictus"],
        "diffusion_radius": 2,
        "diffusion_factor": 0.4,    # 1-hop: 40%, 2-hop: 16% (factor^2)
        "units_per_location": 1.5,
        "km_per_unit": 0.4,         # fogging covers less distance per unit due to density
        "recommended_when": "Permethrin_Resistant == 1 OR Albopictus dominant",
        "description": "Broad-spectrum thermal fogging for dense outbreaks and resistant strains."
    },
    "Temephos Larvicide": {
        "targets": ["larvae (any species)"],
        "diffusion_radius": 0,      # treats only the exact node (water bodies)
        "diffusion_factor": 0.0,
        "units_per_location": 0.75,
        "km_per_unit": 0.8,         # concentrated formula; covers more distance per unit
        "recommended_when": "Water_Index > 0.7",
        "description": "Granular larvicide applied directly into stagnant water reservoirs."
    }
}
