from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx

from data_generator import (
    build_city_graph,
    generate_location_data,
    generate_symptom_data,
    generate_hospital_inventory,
    MEDICINES,
    ZONE_MAP
)
import ai_models
import optimization

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    
st.set_page_config(
    page_title="MosquitoSafe BD | AI Mosquito Intelligence",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_and_train_system():

    G, node_lookup, name_to_id = build_city_graph()

    df_locations = generate_location_data(G)
    df_symptoms = generate_symptom_data()


    scaled_env, scaler = ai_models.preprocess_data(df_locations)
    outbreak_model = ai_models.train_outbreak_model(df_locations, scaler)
    symptom_model = ai_models.train_symptom_model(df_symptoms)
    knn_model = ai_models.train_knn(df_locations)
    df_with_risk, label_map, kmeans_model = ai_models.cluster_locations(df_locations)

    return {
        "G": G,
        "node_lookup": node_lookup,
        "name_to_id": name_to_id,
        "df_locations": df_locations,
        "df_symptoms": df_symptoms,
        "scaler": scaler,
        "outbreak_model": outbreak_model,
        "symptom_model": symptom_model,
        "knn_model": knn_model,
        "df_with_risk": df_with_risk,
        "label_map": label_map,
        "kmeans_model": kmeans_model
    }

SYSTEM = load_and_train_system()
G = SYSTEM["G"]
node_lookup = SYSTEM["node_lookup"]
name_to_id = SYSTEM["name_to_id"]
df_with_risk = SYSTEM["df_with_risk"]
scaler = SYSTEM["scaler"]
outbreak_model = SYSTEM["outbreak_model"]
symptom_model = SYSTEM["symptom_model"]
knn_model = SYSTEM["knn_model"]

if "hospital_inventory" not in st.session_state:
    st.session_state["hospital_inventory"] = generate_hospital_inventory()

if "delivery_orders" not in st.session_state:
    st.session_state["delivery_orders"] = []

CITIZEN_ID = "citizen_1"
CITIZEN_NAME = "Demo Citizen"


def plot_city_network(
    highlight_path: List[str] = None,
    spray_stops: List[str] = None,
    coverage_map: Dict[str, float] = None,
    title: str = "Dhaka City Mosquito & Healthcare Network"
):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes}

    risk_colors = {
        "High": "#e74c3c", 
        "Medium": "#f39c12", 
        "Low": "#2ecc71"  
    }

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#d2d7d9", width=1.4, alpha=0.7)
    edge_labels = {(u, v): f"{d['weight']}m" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=6.5, font_color="#7f8c8d")

    is_spray = spray_stops is not None
    if highlight_path and len(highlight_path) > 1:
        path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
        if is_spray:
            nx.draw_networkx_edges(
                G, pos, edgelist=path_edges, ax=ax,
                edge_color="#8e44ad", width=4.2, alpha=0.9
            )
        else:
            nx.draw_networkx_edges(
                G, pos, edgelist=path_edges, ax=ax,
                edge_color="#0984e3", width=5.0, alpha=0.95
            )

    node_color_list = []
    node_size_list = []
    risk_lookup = dict(zip(df_with_risk["Node_ID"], df_with_risk["Risk_Level"]))

    for n in G.nodes:
        if G.nodes[n].get("is_hospital", False):
            node_color_list.append("#c0392b")
            node_size_list.append(270)
        elif n == "DEPOT":
            node_color_list.append("#2c3e50")
            node_size_list.append(300)
        elif coverage_map and n in coverage_map:
            cov = coverage_map[n]
            if cov >= 0.8:
                node_color_list.append("#27ae60")
            elif cov >= 0.4:
                node_color_list.append("#f39c12")
            else:
                node_color_list.append(risk_colors.get(risk_lookup.get(n, "Low"), "#95a5a6"))
            node_size_list.append(240)
        else:
            r = risk_lookup.get(n, "Low")
            node_color_list.append(risk_colors.get(r, "#95a5a6"))
            node_size_list.append(240)

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_color_list,
        node_size=node_size_list,
        edgecolors="#2c3e50",
        linewidths=1.0
    )

    if highlight_path and not is_spray:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[highlight_path[0]], ax=ax,
            node_color="#00b894", node_size=310, edgecolors="#006266", linewidths=2.2
        )

        nx.draw_networkx_nodes(
            G, pos, nodelist=[highlight_path[-1]], ax=ax,
            node_color="#d63031", node_size=320, edgecolors="#740001", linewidths=2.2
        )

    if spray_stops:

        nx.draw_networkx_nodes(
            G, pos, nodelist=spray_stops, ax=ax,
            node_color="#9b59b6", node_size=290, edgecolors="#4a235a", linewidths=1.8
        )

    labels = {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7.5, font_weight="bold", font_color="#2d3436")

    legend_elements = [
        mpatches.Patch(color="#e74c3c", label="High Risk / Hospital"),
        mpatches.Patch(color="#f39c12", label="Medium Risk / Partial Treatment"),
        mpatches.Patch(color="#2ecc71", label="Low Risk Node"),
        mpatches.Patch(color="#2c3e50", label="Vector Depot")
    ]

    if highlight_path and not is_spray:
        legend_elements.append(
            mlines.Line2D([0], [0], color="#0984e3", lw=3.5, label="Emergency Ambulance Route")
        )
        legend_elements.append(mpatches.Patch(color="#00b894", label="Citizen Origin (Start)"))
        legend_elements.append(mpatches.Patch(color="#d63031", label="Destination Hospital (End)"))
    elif is_spray:
        legend_elements.append(
            mlines.Line2D([0], [0], color="#8e44ad", lw=3.5, label="Optimized Spray Tour")
        )
        legend_elements.append(mpatches.Patch(color="#9b59b6", label="Chemical Spray Stop"))

    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=14)
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/mosquito.png", width=64)
    st.title("MosquitoSafe BD")
    st.caption("AI-Powered Urban Vector & Health Intelligence • Dhaka")
    st.markdown("---")

    role = st.sidebar.radio(
        "I am:",
        ["Citizen", "Health Official", "Hospital Staff"]
    )

    st.markdown("---")
    
    st.caption("Deterministic Simulation • Dhaka Municipal Zones")

