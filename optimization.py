#contains algorithms
from typing import List, Dict, Tuple, Any
import math
import random
import networkx as nx
import constraint

from data_generator import MEDICINES


def _euclidean_heuristic(u: str, v: str, G: nx.Graph) -> float:
    node_u = G.nodes[u]
    node_v = G.nodes[v]
    d_lat = node_u["lat"] - node_v["lat"]
    d_lon = node_u["lon"] - node_v["lon"]
    degree_dist = math.sqrt(d_lat * d_lat + d_lon * d_lon)
    return degree_dist * 1000.0


def astar_route(G: nx.Graph, start_id: str, end_id: str) -> Tuple[List[str], float]:
    if start_id not in G or end_id not in G:
        return [], float("inf")

    if start_id == end_id:
        return [start_id], 0.0

    try:
        heuristic = lambda u, v: _euclidean_heuristic(u, v, G)
        path = nx.astar_path(G, source=start_id, target=end_id, heuristic=heuristic, weight="weight")
        cost = nx.astar_path_length(G, source=start_id, target=end_id, heuristic=heuristic, weight="weight")
        return path, float(cost)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], float("inf")


def find_nearest_hospital(
    G: nx.Graph,
    from_node_id: str,
    hospital_inventory: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    candidate_hospitals = list(hospital_inventory.keys())
    if not candidate_hospitals or from_node_id not in G:
        return {
            "hospital": None,
            "path": [],
            "travel_time": None,
            "relaxed": False,
            "active_constraints": [],
            "excluded": [],
            "reason": "Invalid origin or empty hospital inventory."
        }

    problem_strict = constraint.Problem()
    problem_strict.addVariable("hospital", candidate_hospitals)
    problem_strict.addConstraint(lambda h: hospital_inventory[h]["Available_Beds"] > 0, ["hospital"])
    problem_strict.addConstraint(lambda h: hospital_inventory[h]["Has_Specialist"] is True, ["hospital"])
    strict_solutions = problem_strict.getSolutions()

    active_constraints = ["Available_Beds > 0", "Has_Specialist == True"]
    relaxed = False
    valid_candidates = [sol["hospital"] for sol in strict_solutions]

    if not valid_candidates:
        relaxed = True
        active_constraints = ["Available_Beds > 0"]
        problem_relaxed = constraint.Problem()
        problem_relaxed.addVariable("hospital", candidate_hospitals)
        problem_relaxed.addConstraint(lambda h: hospital_inventory[h]["Available_Beds"] > 0, ["hospital"])
        relaxed_solutions = problem_relaxed.getSolutions()
        valid_candidates = [sol["hospital"] for sol in relaxed_solutions]

    excluded: List[str] = []
    for h_id in candidate_hospitals:
        h = hospital_inventory[h_id]
        if h["Available_Beds"] <= 0:
            excluded.append(f"{h['name']}: no beds available")
        elif not relaxed and not h["Has_Specialist"]:
            excluded.append(f"{h['name']}: no dengue specialist on duty")

    if not valid_candidates:
        return {
            "hospital": None,
            "path": [],
            "travel_time": None,
            "relaxed": relaxed,
            "active_constraints": active_constraints,
            "excluded": excluded,
            "reason": "No hospital has available beds right now."
        }

    best_hospital_id = None
    best_path: List[str] = []
    min_travel_time = float("inf")

    for h_id in valid_candidates:
        path, travel_time = astar_route(G, from_node_id, h_id)
        if travel_time < min_travel_time:
            min_travel_time = travel_time
            best_path = path
            best_hospital_id = h_id

    if best_hospital_id is None:
        return {
            "hospital": None,
            "path": [],
            "travel_time": None,
            "relaxed": relaxed,
            "active_constraints": active_constraints,
            "excluded": excluded,
            "reason": "No reachable hospital found from this location."
        }

    return {
        "hospital": hospital_inventory[best_hospital_id],
        "path": best_path,
        "travel_time": min_travel_time,
        "relaxed": relaxed,
        "active_constraints": active_constraints,
        "excluded": excluded,
        "reason": "Optimal hospital found." if not relaxed else "Note: No specialist available; nearest hospital with open beds shown instead."
    }


def plan_spray_route(
    G: nx.Graph,
    df_with_risk: Any,
    start_node_id: str,
    medicine_name: str,
    units_available: float
) -> Dict[str, Any]:
    random.seed(42)

    med_spec = MEDICINES.get(medicine_name, MEDICINES["Permethrin Spray"])
    km_per_unit = float(med_spec["km_per_unit"])
    radius = int(med_spec["diffusion_radius"])
    factor = float(med_spec["diffusion_factor"])
    distance_budget = units_available * km_per_unit

    RISK_WEIGHTS = {"High": 3.0, "Medium": 1.5, "Low": 0.5}
    node_risk_map = dict(zip(df_with_risk["Node_ID"], df_with_risk["Risk_Level"]))
    infrastructure_ids = {
        n for n in G.nodes
        if G.nodes[n].get("is_hospital", False) or n == "DEPOT"
    }
    all_high = [nid for nid, r in node_risk_map.items() if r == "High"]
    total_high_risk = len(all_high)
    trivial = [start_node_id]
    depth_limit = min(len(G.nodes), 8)

    def risk_weight(node_id: str) -> float:
        if node_id in infrastructure_ids:
            return 0.0
        return RISK_WEIGHTS.get(node_risk_map.get(node_id, "Low"), 0.5)

    def path_distance(p: List[str]) -> float:
        return sum(G[u][v]["weight"] for u, v in zip(p[:-1], p[1:]))

    candidate_paths: List[List[str]] = [trivial]

    def _dfs(path: List[str], dist_so_far: float) -> None:
        if len(path) - 1 >= depth_limit:
            return
        current = path[-1]
        for nbr in G.neighbors(current):
            if nbr in path:
                continue
            edge_w = float(G[current][nbr]["weight"])
            new_dist = dist_so_far + edge_w
            if new_dist > distance_budget:
                continue
            path.append(nbr)
            candidate_paths.append(list(path))
            _dfs(path, new_dist)
            path.pop()

    if start_node_id in G:
        _dfs([start_node_id], 0.0)

    scored: List[Tuple[float, float, List[str]]] = []
    for p in candidate_paths:
        scored.append((sum(risk_weight(n) for n in p[1:]), path_distance(p), p))

    if not scored:
        scored = [(0.0, 0.0, trivial)]

    max_score = max(s for s, _, _ in scored)
    top_by_score = [(s, d, p) for s, d, p in scored if s == max_score]
    min_dist = min(d for _, d, _ in top_by_score)
    top_by_dist = [(s, d, p) for s, d, p in top_by_score if d == min_dist]

    was_random_tiebreak = len(top_by_dist) > 1
    if len(top_by_dist) == 0:
        best_score, best_dist, best_path = 0.0, 0.0, trivial
        was_random_tiebreak = False
    else:
        best_score, best_dist, best_path = random.choice(top_by_dist)

    coverage: Dict[str, float] = {node: 0.0 for node in G.nodes}
    spray_stops = best_path[1:]

    for sprayed_node in spray_stops:
        hops = nx.single_source_shortest_path_length(G, sprayed_node, cutoff=radius)
        for n, d in hops.items():
            if radius == 0:
                dose = 1.0 if d == 0 else 0.0
            else:
                dose = 1.0 * (factor ** d)
            coverage[n] = min(1.0, coverage[n] + dose)

    units_used = best_dist / km_per_unit if km_per_unit > 0 else 0.0
    units_remaining = max(0.0, units_available - units_used)
    high_risk_covered = sum(1 for nid in all_high if coverage.get(nid, 0.0) >= 0.50)
    path_names = [G.nodes[n].get("name", n) for n in best_path]

    result: Dict[str, Any] = {
        "path": best_path,
        "path_names": path_names,
        "score": round(best_score, 2),
        "total_distance_km": round(best_dist, 2),
        "units_used": round(units_used, 2),
        "units_remaining": round(units_remaining, 2),
        "coverage_map": coverage,
        "high_risk_covered": high_risk_covered,
        "total_high_risk": total_high_risk,
        "was_random_tiebreak": was_random_tiebreak,
        "route_node_ids": best_path,
        "stops": spray_stops,
    }

    if best_path == trivial:
        result["reason"] = "Distance budget too small to reach any adjacent node."
    else:
        has_high_or_med = any(
            node_risk_map.get(n) in ("High", "Medium") for n in best_path[1:]
        )
        if not has_high_or_med:
            result["note"] = "No High or Medium risk nodes within range."

    return result
