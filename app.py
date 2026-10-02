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

if role == "Citizen":
    st.title("Citizen Health & Emergency Triage Portal")
    st.markdown(
        "Check symptoms, monitor local mosquito risk in your neighborhood, and navigate to the "
        "nearest available emergency medical facility with verified open beds."
    )

    regular_locations = df_with_risk["Location"].tolist()
    
    col_loc1, col_loc2 = st.columns([2, 1])
    with col_loc1:
        chosen_loc_name = st.selectbox(
            "Select Your Location in Dhaka:",
            regular_locations,
            index=0
        )
    
    curr_row = df_with_risk[df_with_risk["Location"] == chosen_loc_name].iloc[0]
    curr_node_id = curr_row["Node_ID"]
    curr_zone = curr_row["Zone"]
    curr_risk = curr_row["Risk_Level"]

    with col_loc2:
        st.markdown(f"**Zone:** `{curr_zone}` (Node `{curr_node_id}`)")
        if curr_risk == "High":
            st.error(f"Local Mosquito Risk: **{curr_risk}**")
        elif curr_risk == "Medium":
            st.warning(f"Local Mosquito Risk: **{curr_risk}**")
        else:
            st.success(f"Local Mosquito Risk: **{curr_risk}**")

    st.markdown("---")

    tab_symptom, tab_hospital = st.tabs(["🩺 Dengue / Chikungunya Symptom Checker", "🏥 Nearest Emergency Hospital Finder"])

    with tab_symptom:
        st.subheader("Diagnostic AI Symptom Triage (Logistic Regression)")
        st.write("Indicate your current physical symptoms for real-time risk assessment:")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sym_fever = st.checkbox("Acute Sudden Fever", value=True)
            sym_joint = st.checkbox("Severe Joint / Muscle Pain", value=False)
        with col_s2:
            sym_rash = st.checkbox("Skin Rash / Petechiae", value=False)
            sym_headache = st.checkbox("Severe Frontal Headache", value=False)
        with col_s3:
            sym_eye = st.checkbox("Retro-Orbital (Behind-the-Eye) Pain", value=False)

        symptoms = {
            "Fever": int(sym_fever),
            "JointPain": int(sym_joint),
            "Rash": int(sym_rash),
            "Headache": int(sym_headache),
            "EyePain": int(sym_eye),
        }
        symptom_key = f"{curr_node_id}|{symptoms}"

        if st.button("Analyze My Symptoms", type="primary"):
            prob, rule_note = ai_models.predict_symptom_risk(symptoms, curr_risk, symptom_model)
            st.session_state["symptom_result"] = {
                "prob": prob,
                "note": rule_note,
                "key": symptom_key,
                "zone": curr_zone,
                "risk": curr_risk,
            }

        if "symptom_result" in st.session_state and st.session_state["symptom_result"]["key"] == symptom_key:
            sr = st.session_state["symptom_result"]
            prob = sr["prob"]
            st.write(f"### Infection Likelihood: **{prob * 100:.1f}%**")
            st.caption(f"Rule applied: {sr['note']} | Zone `{sr['zone']}` ({sr['risk']} mosquito risk) adds local exposure context.")
            st.progress(float(prob))

            if prob >= 0.65:
                st.error(
                    "**HIGH RISK**: Clinical symptoms strongly correlate with acute Dengue/Chikungunya infection! "
                    "Please proceed immediately to an emergency hospital. Maintain oral hydration and avoid NSAIDs."
                )
            elif prob >= 0.35:
                st.warning(
                    "**MODERATE RISK**: Symptoms present moderate possibility of vector-borne illness. "
                    "Consult a physician if fever persists beyond 24 hours."
                )
            else:
                st.success(
                    "**LOW RISK**: Symptoms do not immediately indicate acute mosquito-borne illness. "
                    "Monitor your condition and rest well."
                )

    with tab_hospital:
        st.subheader("Emergency Hospital Routing (CSP + A* Search)")
        st.write(
            "Uses **Constraint Satisfaction Problem (CSP)** to verify available beds and on-duty specialists, "
            "then runs **A* Search** on the road network to navigate you there in minimal time."
        )

        if st.button("Find & Route to Nearest Available Hospital", type="primary"):
            st.session_state["citizen_route"] = optimization.find_nearest_hospital(
                G, curr_node_id, st.session_state["hospital_inventory"]
            )
            st.session_state["citizen_route_origin"] = curr_node_id

        if "citizen_route" in st.session_state and st.session_state.get("citizen_route_origin") == curr_node_id:
            res = st.session_state["citizen_route"]

            if res["hospital"] is None:
                st.error(f"{res['reason']}")
            else:
                h = res["hospital"]
                time_min = res["travel_time"]
                path = res["path"]
                relaxed = res["relaxed"]

                if res.get("excluded"):
                    with st.expander("ℹHospitals skipped (no beds or no dengue care)"):
                        for line in res["excluded"]:
                            st.write(f"- {line}")

                if relaxed:
                    st.warning(
                        "**Constraint Relaxed**: No hospital with both open beds and a dengue specialist is available. "
                        "Routing to the nearest hospital with open beds instead."
                    )
                else:
                    st.success("**Optimal Hospital Found**: Open beds and dengue specialist confirmed!")

                col_h1, col_h2, col_h3 = st.columns(3)
                with col_h1:
                    st.metric("Destination Hospital", h["name"])
                    st.caption(f"Zone: {h['zone']} (Node {h['node']})")
                with col_h2:
                    st.metric("Estimated Travel Time", f"{time_min:.1f} mins")
                    st.caption(f"Road distance: {len(path)-1} segments")
                with col_h3:
                    st.metric("Available Beds", f"{h['Available_Beds']} / {h['Total_Beds']}")
                    spec_txt = "👨‍⚕️ Yes (On Duty)" if h["Has_Specialist"] else "No Specialist"
                    st.caption(f"Specialist: {spec_txt}")

                path_names = [f"`{node_lookup[n]}` ({n})" for n in path]
                st.markdown(f"**Turn-by-Turn Route:** {' ➔ '.join(path_names)}")

                fig = plot_city_network(
                    highlight_path=path,
                    title=f"Emergency Route: {chosen_loc_name} ({curr_node_id}) ➔ {h['name']} ({h['node']})"
                )
                st.pyplot(fig)

                st.markdown("---")
                st.subheader("Order Medicine / IV Fluid for Home Delivery")
                h_id = h["node"]
                stock_items = list(st.session_state["hospital_inventory"][h_id]["stock"].keys())
                col_o1, col_o2 = st.columns(2)
                with col_o1:
                    order_item = st.selectbox("Item to order:", stock_items, key="citizen_order_item")
                with col_o2:
                    order_qty = st.number_input("Quantity:", min_value=1, max_value=20, value=1, step=1, key="citizen_order_qty")

                if st.button("Place Delivery Order", key="citizen_place_order"):
                    st.session_state["delivery_orders"].append({
                        "id": len(st.session_state["delivery_orders"]) + 1,
                        "citizen_id": CITIZEN_ID,
                        "citizen_name": CITIZEN_NAME,
                        "citizen_location": chosen_loc_name,
                        "hospital_id": h_id,
                        "hospital_name": h["name"],
                        "item": order_item,
                        "quantity": int(order_qty),
                        "status": "pending",
                    })
                    st.success(f"Order placed: {order_qty} × {order_item} from {h['name']}. Awaiting hospital delivery.")

                my_orders = [
                    o for o in st.session_state["delivery_orders"]
                    if o["citizen_id"] == CITIZEN_ID
                ]
                if my_orders:
                    st.markdown("**Your orders:**")
                    for o in my_orders:
                        status_icon = "Delivered" if o["status"] == "delivered" else "⏳ Pending"
                        st.write(f"- #{o['id']}: {o['quantity']} X {o['item']} from {o['hospital_name']} — {status_icon}")
                        if o["status"] == "delivered":
                            st.success(f"Delivery #{o['id']} completed successfully!")

