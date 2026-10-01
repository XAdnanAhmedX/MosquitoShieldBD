from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import networkx as nx
MEDICINES: Dict[str, Dict[str, Any]] = {
    "Permethrin Spray": {
        "targets": ["Aedes Aegypti"],
        "diffusion_radius": 1,      
        "diffusion_factor": 0.5,    
        "units_per_location": 1.0,  
        "km_per_unit": 0.6,         
        "recommended_when": "Permethrin_Resistant == 0",
        "description": "Standard adulticide targeting adult Aedes Aegypti mosquitoes."
    },
    "Malathion Fog": {
        "targets": ["Aedes Aegypti", "Aedes Albopictus"],
        "diffusion_radius": 2,
        "diffusion_factor": 0.4,    
        "units_per_location": 1.5,
        "km_per_unit": 0.4,         
        "recommended_when": "Permethrin_Resistant == 1 OR Albopictus dominant",
        "description": "Broad-spectrum thermal fogging for dense outbreaks and resistant strains."
    },
    "Temephos Larvicide": {
        "targets": ["larvae (any species)"],
        "diffusion_radius": 0,      
        "diffusion_factor": 0.0,
        "units_per_location": 0.75,
        "km_per_unit": 0.8,         
        "recommended_when": "Water_Index > 0.7",
        "description": "Granular larvicide applied directly into stagnant water reservoirs."
    }
}
def build_city_graph() -> Tuple[nx.Graph, Dict[str, str], Dict[str, str]]:
    """
    Constructs the hand-defined Dhaka city graph.
    
    Returns:
        G: networkx.Graph with node attributes (name, zone, lat, lon, is_hospital)
           and edge attributes (weight = travel minutes).
        node_lookup: dict mapping node_id -> human-readable name.
        name_to_id: dict mapping human-readable name -> node_id.
    """
    G = nx.Graph()

    # 1. Define nodes: (id, name, zone, lat, lon, is_hospital)
    raw_nodes = [
        # DHANMONDI
        ("D1", "Dhanmondi 2", "Dhanmondi", 23.7461, 90.3742, False),
        ("D2", "Dhanmondi 8", "Dhanmondi", 23.7489, 90.3720, False),
        ("D3", "Dhanmondi 15", "Dhanmondi", 23.7512, 90.3698, False),
        ("D4", "Satmasjid Road", "Dhanmondi", 23.7535, 90.3760, False),
        ("D5", "Dhanmondi 27", "Dhanmondi", 23.7448, 90.3705, False),
        ("D6", "Dhanmondi Lake", "Dhanmondi", 23.7503, 90.3741, False),
        ("H1", "Popular Hospital", "Dhanmondi", 23.7475, 90.3731, True),

        # SHAHBAGH
        ("S1", "Shahbagh Circle", "Shahbagh", 23.7392, 90.3950, False),
        ("S2", "TSC", "Shahbagh", 23.7338, 90.3969, False),
        ("S3", "Ramna Park", "Shahbagh", 23.7361, 90.4003, False),
        ("S4", "Elephant Road", "Shahbagh", 23.7420, 90.3930, False),
        ("S5", "Nilkhet", "Shahbagh", 23.7356, 90.3912, False),
        ("H2", "BSMMU Hospital", "Shahbagh", 23.7403, 90.3958, True),

        # HAZARIBAGH
        ("Z1", "Hazaribagh Bus Stand", "Hazaribagh", 23.7205, 90.3600, False),
        ("Z2", "Jheel Road", "Hazaribagh", 23.7180, 90.3575, False),
        ("Z3", "Ekuria", "Hazaribagh", 23.7225, 90.3555, False),
        ("Z4", "Rayerbazar", "Hazaribagh", 23.7245, 90.3620, False),
        ("Z5", "Dhal Kandia", "Hazaribagh", 23.7195, 90.3640, False),
        ("H3", "Hazaribagh Clinic", "Hazaribagh", 23.7210, 90.3608, True),

        # ZIGATOLA
        ("G1", "Zigatola Bus Stand", "Zigatola", 23.7290, 90.3680, False),
        ("G2", "Lalmatia A Block", "Zigatola", 23.7310, 90.3660, False),
        ("G3", "Lalmatia B Block", "Zigatola", 23.7325, 90.3645, False),
        ("G4", "Lalmatia C Block", "Zigatola", 23.7340, 90.3630, False),
        ("G5", "Jigatola Road", "Zigatola", 23.7275, 90.3695, False),
        ("H4", "Ibn Sina Zigatola", "Zigatola", 23.7318, 90.3652, True),

        # KALABAGAN (No hospital in Kalabagan)
        ("K1", "Kalabagan Market", "Kalabagan", 23.7520, 90.3800, False),
        ("K2", "Panthapath", "Kalabagan", 23.7545, 90.3835, False),
        ("K3", "Green Road", "Kalabagan", 23.7560, 90.3815, False),
        ("K4", "Mirpur Road Jct", "Kalabagan", 23.7498, 90.3778, False),
        ("K5", "Free School Street", "Kalabagan", 23.7535, 90.3860, False),

        # VECTOR CONTROL DEPOT (Starting location for spray units)
        ("DEPOT", "Vector Control Depot", "Depot", 23.7300, 90.3700, False)
    ]

    for node_id, name, zone, lat, lon, is_hospital in raw_nodes:
        G.add_node(
            node_id,
            name=name,
            zone=zone,
            lat=lat,
            lon=lon,
            is_hospital=is_hospital
        )

    # 2. Define edges with travel times in minutes
    raw_edges = [
        # Within Dhanmondi
        ("D1", "D2", 3), ("D2", "D3", 4), ("D3", "D4", 3),
        ("D4", "D6", 4), ("D5", "D6", 3), ("D1", "D5", 4),
        ("D1", "H1", 2), ("D2", "H1", 2),

        # Within Shahbagh
        ("S1", "S2", 5), ("S2", "S3", 4), ("S3", "S4", 6),
        ("S4", "S5", 4), ("S5", "S2", 5),
        ("S1", "H2", 2), ("S4", "H2", 3),

        # Within Hazaribagh
        ("Z1", "Z2", 5), ("Z2", "Z3", 4), ("Z3", "Z4", 6),
        ("Z4", "Z5", 4), ("Z5", "Z1", 5),
        ("Z1", "H3", 2), ("Z4", "H3", 6),

        # Within Zigatola
        ("G1", "G2", 4), ("G2", "G3", 3), ("G3", "G4", 3),
        ("G4", "G5", 5), ("G5", "G1", 4),
        ("G3", "H4", 2), ("G2", "H4", 3),

        # Within Kalabagan
        ("K1", "K2", 5), ("K2", "K3", 4), ("K3", "K5", 4),
        ("K4", "K5", 5), ("K4", "K1", 4),

        # Cross-zone roads
        ("D5", "K4", 6),    # Dhanmondi 27 -> Mirpur Road Jct
        ("D6", "S1", 7),    # Dhanmondi Lake -> Shahbagh Circle
        ("K1", "G1", 6),    # Kalabagan Market -> Zigatola Bus Stand
        ("G1", "Z5", 7),    # Zigatola Bus Stand -> Dhal Kandia
        ("Z3", "D4", 9),    # Ekuria -> Satmasjid Road
        ("S5", "K2", 5),    # Nilkhet -> Panthapath
        ("S4", "K3", 5),    # Elephant Road -> Green Road
        ("K1", "D5", 5),    # Kalabagan Market -> Dhanmondi 27
        ("DEPOT", "G1", 5), # Depot -> Zigatola Bus Stand
        ("DEPOT", "Z5", 6), # Depot -> Dhal Kandia
    ]

    for u, v, weight in raw_edges:
        G.add_edge(u, v, weight=weight)

