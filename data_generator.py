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

# 3. Critical Verification: Ensure graph is fully connected
    if not nx.is_connected(G):
        raise ValueError("Critical Error: The city graph G is not fully connected! Check edge definitions.")

    # Bidirectional lookups
    node_lookup = {n: G.nodes[n]["name"] for n in G.nodes}
    name_to_id = {G.nodes[n]["name"]: n for n in G.nodes}

    return G, node_lookup, name_to_id


# Instantiate once to construct global mappings
_GLOBAL_G, _GLOBAL_NODE_LOOKUP, _GLOBAL_NAME_TO_ID = build_city_graph()

ZONE_MAP: Dict[str, str] = {
    _GLOBAL_NODE_LOOKUP[n]: _GLOBAL_G.nodes[n]["zone"]
    for n in _GLOBAL_G.nodes
}



# SYNTHETIC DATA GENERATORS (DETERMINISTIC)

def generate_location_data(G: nx.Graph) -> pd.DataFrame:
    """
    Generates synthetic epidemiological and environmental features for all
    non-hospital, non-depot location nodes.
    
    Guarantees deterministic results using np.random.seed(42).
    """
    np.random.seed(42)
    rows = []

    # Filter out hospital and depot nodes
    regular_nodes = [
        n for n in G.nodes
        if not G.nodes[n].get("is_hospital", False) and G.nodes[n].get("zone") != "Depot"
    ]

    # Pre-generate values based on zone characteristics
    for node_id in regular_nodes:
        name = G.nodes[node_id]["name"]
        zone = G.nodes[node_id]["zone"]

        if zone in ["Hazaribagh", "Zigatola"]:
            # Tightly packed, canal-adjacent, drainage challenges
            rainfall = np.random.uniform(170.0, 260.0)
            water_index = np.random.uniform(0.60, 0.95)
            urban_score = np.random.uniform(0.25, 0.55)
            temperature = np.random.uniform(28.0, 33.5)
            humidity = np.random.uniform(75.0, 92.0)
        elif zone in ["Dhanmondi", "Shahbagh"]:
            # Highly urbanized, residential/institutional, paved
            rainfall = np.random.uniform(80.0, 150.0)
            water_index = np.random.uniform(0.20, 0.50)
            urban_score = np.random.uniform(0.70, 0.96)
            temperature = np.random.uniform(30.0, 35.5)
            humidity = np.random.uniform(62.0, 78.0)
        else:
            # Kalabagan (mixed commercial/dense residential)
            rainfall = np.random.uniform(110.0, 190.0)
            water_index = np.random.uniform(0.35, 0.70)
            urban_score = np.random.uniform(0.45, 0.75)
            temperature = np.random.uniform(29.0, 34.0)
            humidity = np.random.uniform(68.0, 84.0)

        # Vector counts scale logically
        # Aegypti breeds in artificial containers in urban areas
        aegypti = int(35 + 85 * urban_score + np.random.normal(0, 5))
        aegypti = max(10, aegypti)

        # Albopictus prefers vegetation and natural water collections
        albopictus = int(25 + 95 * water_index + np.random.normal(0, 6))
        albopictus = max(5, albopictus)

        # Actual dengue/chikungunya cases formula with environmental drivers
        noise = np.random.normal(0, 6)
        raw_cases = 0.35 * rainfall + 0.45 * temperature - 0.22 * humidity + noise
        actual_cases = int(np.clip(raw_cases, 0, 150))

        rows.append({
            "Node_ID": node_id,
            "Location": name,
            "Zone": zone,
            "Temperature": round(float(temperature), 1),
            "Rainfall": round(float(rainfall), 1),
            "Humidity": round(float(humidity), 1),
            "Aegypti_Count": int(aegypti),
            "Albopictus_Count": int(albopictus),
            "Water_Index": round(float(water_index), 2),
            "Urbanization_Score": round(float(urban_score), 2),
            "Actual_Cases": int(actual_cases)
        })

    df = pd.DataFrame(rows)

    # Permethrin resistance heuristic:
    # High aegypti density (> median) combined with standing stagnant water (> 0.5)
    med_aegypti = df["Aegypti_Count"].median()
    df["Permethrin_Resistant"] = (
        (df["Aegypti_Count"] > med_aegypti) & (df["Water_Index"] > 0.50)
    ).astype(int)

    return df


def generate_symptom_data() -> pd.DataFrame:
    """
    Generates 200 synthetic clinical records of patient symptoms and disease diagnosis.
    
    Clinical Rules:
    - Fever + JointPain + EyePain -> 90% chance Has_Disease=1 (classic dengue triad)
    - No Fever -> Has_Disease=0 always
    - Otherwise -> 30% baseline chance
    """
    np.random.seed(42)
    n_samples = 200

    # 1/0 binary flags for symptoms
    fevers = np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75])
    joint_pains = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    rashes = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    headaches = np.random.choice([0, 1], size=n_samples, p=[0.35, 0.65])
    eye_pains = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])

    has_disease = []
    for f, j, r, h, e in zip(fevers, joint_pains, rashes, headaches, eye_pains):
        if f == 0:
            # Dengue/Chikungunya almost always begins with acute fever
            has_disease.append(0)
        elif f == 1 and j == 1 and e == 1:
            # Retro-orbital pain + fever + severe arthralgia = high probability
            has_disease.append(1 if np.random.rand() < 0.90 else 0)
        else:
            has_disease.append(1 if np.random.rand() < 0.30 else 0)

    df_symptoms = pd.DataFrame({
        "Fever": fevers,
        "JointPain": joint_pains,
        "Rash": rashes,
        "Headache": headaches,
        "EyePain": eye_pains,
        "Has_Disease": has_disease
    })
    return df_symptoms

def generate_hospital_inventory() -> Dict[str, Dict[str, Any]]:
    """
    Returns initial mutable hospital dictionary for st.session_state.
    """
    return {
        "H1": {
            "name": "Popular Hospital",
            "node": "H1",
            "zone": "Dhanmondi",
            "Total_Beds": 60,
            "Available_Beds": 35,
            "Has_Specialist": True,
            "stock": {"Oral Medication": 80, "IV Saline": 40, "Injection": 30}
        },
        "H2": {
            "name": "BSMMU Hospital",
            "node": "H2",
            "zone": "Shahbagh",
            "Total_Beds": 120,
            "Available_Beds": 55,
            "Has_Specialist": True,
            "stock": {"Oral Medication": 150, "IV Saline": 90, "Injection": 60}
        },
        "H3": {
            "name": "Hazaribagh Clinic",
            "node": "H3",
            "zone": "Hazaribagh",
            "Total_Beds": 30,
            "Available_Beds": 8,
            "Has_Specialist": False,
            "stock": {"Oral Medication": 40, "IV Saline": 15, "Injection": 10}
        },
        "H4": {
            "name": "Ibn Sina Zigatola",
            "node": "H4",
            "zone": "Zigatola",
            "Total_Beds": 50,
            "Available_Beds": 22,
            "Has_Specialist": True,
            "stock": {"Oral Medication": 70, "IV Saline": 35, "Injection": 25}
        },
    }



