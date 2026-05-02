#!/usr/bin/env python3
"""
plot_cloud_optimizers.py — Visualisation de la comparaison COBYLA vs SPSA
=========================================================================
Auteur  : Boussaiah Younes (INPT - Cloud & IoT)
Date    : Avril 2026
Objectif: Tracer les courbes de convergence de COBYLA et SPSA
          à partir des résultats générés par analyze_cloud_optimizers.py.
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "vqe_cloud_optimizers_comparison.csv"
JSON_PATH = RESULTS_DIR / "vqe_cloud_optimizers_summary.json"
OUTPUT_PATH = RESULTS_DIR / "vqe_cloud_optimizers_comparison.png"

# Couleurs
COLORS = {
    "COBYLA": "#1192E8",  # Cyan IBM
    "SPSA": "#9F1853",    # Magenta IBM
}
EXACT_COLOR = "#DA1E28"
BG_COLOR = "#0F0F23"
GRID_COLOR = "#2A2A4A"
TEXT_COLOR = "#E8E8F0"

def load_data():
    if not CSV_PATH.exists():
        print(f"❌ Fichier introuvable : {CSV_PATH}")
        sys.exit(1)

    data = defaultdict(lambda: {"iterations": [], "energies": []})
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opt = row["optimizer"]
            data[opt]["iterations"].append(int(row["iteration"]))
            data[opt]["energies"].append(float(row["energy"]))

    summary = {}
    if JSON_PATH.exists():
        with open(JSON_PATH, "r") as f:
            summary = json.load(f)

    return data, summary

def create_plot(data, summary):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    exact_energy = summary.get("exact_energy")

    for opt_name, opt_data in data.items():
        color = COLORS.get(opt_name, "#FFFFFF")
        ax.plot(
            opt_data["iterations"], opt_data["energies"],
            color=color, linewidth=2, alpha=0.9,
            label=f"{opt_name} (Énergie finale: {opt_data['energies'][-1]:.6f})"
        )
        ax.scatter(
            opt_data["iterations"], opt_data["energies"],
            color=color, s=15, alpha=0.5, edgecolors="none"
        )

    if exact_energy is not None:
        ax.axhline(
            y=exact_energy, color=EXACT_COLOR, linestyle="--",
            linewidth=1.5, alpha=0.8,
            label=f"Énergie exacte : {exact_energy:.6f} Ha"
        )
        ax.axhspan(
            exact_energy - 0.0016, exact_energy + 0.0016,
            color=EXACT_COLOR, alpha=0.08,
            label="Précision chimique (±1.6 mHa)"
        )

    ax.set_xlabel("Nombre d'évaluations", fontsize=14, fontweight="bold", color=TEXT_COLOR, labelpad=12)
    ax.set_ylabel("Énergie (Hartree)", fontsize=14, fontweight="bold", color=TEXT_COLOR, labelpad=12)
    
    backend_name = summary.get("backend", "IBM Quantum")
    ax.set_title(
        rf"VQE H₂ sur {backend_name} — COBYLA vs SPSA",
        fontsize=18, fontweight="bold", color=TEXT_COLOR, pad=20
    )

    ax.grid(True, alpha=0.3, color=GRID_COLOR, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.5)

    ax.legend(
        loc="upper right", fontsize=10, facecolor="#1A1A35",
        edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, framealpha=0.9
    )

    fig.text(
        0.99, 0.01,
        "Boussaiah Younes · INPT · Cloud & IoT",
        fontsize=8, color="#555566", ha="right", va="bottom", style="italic"
    )

    plt.tight_layout()
    return fig

def main():
    print("📊 Visualisation de la comparaison d'optimiseurs")
    data, summary = load_data()
    fig = create_plot(data, summary)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"✅ Graphique sauvegardé : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
