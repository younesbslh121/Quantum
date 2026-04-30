#!/usr/bin/env python3
"""
plot_results.py — Visualisation de la convergence VQE
======================================================
Auteur  : Boussaiah Younes (INPT - Cloud & IoT)
Date    : Avril 2026
Objectif: Tracer la courbe de convergence Énergie vs Itérations
          à partir des résultats générés par run_vqe.py.

Sortie  : results/vqe_convergence.png (300 DPI)
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour la sauvegarde PNG
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── Configuration ──────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "vqe_convergence.csv"
JSON_PATH = RESULTS_DIR / "vqe_summary.json"
OUTPUT_PATH = RESULTS_DIR / "vqe_convergence.png"

# ─── Style ──────────────────────────────────────────────────

# Couleurs inspirées d'IBM Quantum
IBM_PURPLE = "#6929C4"
IBM_CYAN = "#1192E8"
IBM_MAGENTA = "#9F1853"
IBM_TEAL = "#009D9A"
EXACT_COLOR = "#DA1E28"      # Rouge IBM pour l'énergie exacte
BG_COLOR = "#0F0F23"         # Fond sombre
GRID_COLOR = "#2A2A4A"       # Grille subtile
TEXT_COLOR = "#E8E8F0"        # Texte clair


def load_data():
    """Charge les données de convergence et le résumé."""

    # Vérification des fichiers
    if not CSV_PATH.exists():
        print(f"❌ Fichier introuvable : {CSV_PATH}")
        print("   Lancez d'abord : python run_vqe.py")
        sys.exit(1)

    # Lecture du CSV
    iterations = []
    energies = []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row["iteration"]))
            energies.append(float(row["energy"]))

    print(f"📊 Données chargées : {len(iterations)} points")

    # Lecture du JSON (pour l'énergie exacte)
    exact_energy = None
    summary = {}
    if JSON_PATH.exists():
        with open(JSON_PATH, "r") as f:
            summary = json.load(f)
        exact_energy = summary.get("exact_energy_hartree")
        print(f"   Énergie exacte : {exact_energy} Ha")

    return iterations, energies, exact_energy, summary


def create_convergence_plot(iterations, energies, exact_energy, summary):
    """Crée le graphique de convergence avec un style scientifique premium."""

    # ─── Configuration de la figure ──────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # ─── Courbe de convergence ───────────────────────────────
    ax.plot(
        iterations, energies,
        color=IBM_CYAN,
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
        label="Énergie VQE",
    )

    # Points individuels (semi-transparents)
    ax.scatter(
        iterations, energies,
        color=IBM_PURPLE,
        s=8,
        alpha=0.4,
        zorder=4,
        edgecolors="none",
    )

    # Marqueur pour le dernier point (énergie finale)
    if iterations and energies:
        ax.scatter(
            [iterations[-1]], [energies[-1]],
            color=IBM_TEAL,
            s=100,
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
            label=f"Énergie finale : {energies[-1]:.6f} Ha",
        )

    # ─── Ligne de l'énergie exacte ───────────────────────────
    if exact_energy is not None:
        ax.axhline(
            y=exact_energy,
            color=EXACT_COLOR,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
            label=f"Énergie exacte : {exact_energy:.6f} Ha",
        )

        # Zone de précision chimique (±1.6 mHa)
        ax.axhspan(
            exact_energy - 0.0016,
            exact_energy + 0.0016,
            color=EXACT_COLOR,
            alpha=0.08,
            zorder=1,
            label="Précision chimique (±1.6 mHa)",
        )

    # ─── Axes et grille ──────────────────────────────────────
    ax.set_xlabel(
        "Nombre d'évaluations",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        labelpad=12,
    )
    ax.set_ylabel(
        "Énergie (Hartree)",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        labelpad=12,
    )

    # Titre principal
    ax.set_title(
        r"Convergence VQE — Molécule $\mathrm{H_2}$ (STO-3G)",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=20,
    )

    # Sous-titre
    optimizer_name = summary.get("optimizer", "COBYLA")
    total_evals = summary.get("total_evaluations", len(iterations))
    elapsed = summary.get("elapsed_seconds", "?")
    ax.text(
        0.5, 1.02,
        f"Ansatz: TwoLocal (ry, rz + cz) · Optimiseur: {optimizer_name} · "
        f"{total_evals} évaluations · {elapsed}s",
        transform=ax.transAxes,
        fontsize=10,
        color="#888899",
        ha="center",
        va="bottom",
    )

    # Grille
    ax.grid(True, alpha=0.3, color=GRID_COLOR, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=11)

    # Bordures
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.5)

    # ─── Légende ─────────────────────────────────────────────
    legend = ax.legend(
        loc="upper right",
        fontsize=10,
        facecolor="#1A1A35",
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        framealpha=0.9,
    )

    # ─── Annotation : erreur finale ─────────────────────────
    if exact_energy is not None and energies:
        final_error = abs(energies[-1] - exact_energy)
        error_text = f"Δ = {final_error:.2e} Ha"
        chem = "✓ Précision chimique" if final_error < 0.0016 else "✗ Hors précision chimique"

        ax.annotate(
            f"{error_text}\n{chem}",
            xy=(iterations[-1], energies[-1]),
            xytext=(-130, 40),
            textcoords="offset points",
            fontsize=10,
            color=TEXT_COLOR,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#1A1A35",
                edgecolor=IBM_TEAL,
                alpha=0.9,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=IBM_TEAL,
                connectionstyle="arc3,rad=0.2",
            ),
        )

    # ─── Watermark ───────────────────────────────────────────
    fig.text(
        0.99, 0.01,
        "Boussaiah Younes · INPT · Cloud & IoT",
        fontsize=8,
        color="#555566",
        ha="right",
        va="bottom",
        style="italic",
    )

    # ─── Ajustement ──────────────────────────────────────────
    plt.tight_layout()

    return fig


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   📊 VQE H₂ — Visualisation de la convergence          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Charger les données
    iterations, energies, exact_energy, summary = load_data()

    # Créer le graphique
    fig = create_convergence_plot(iterations, energies, exact_energy, summary)

    # Sauvegarder
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)

    print(f"\n✅ Graphique sauvegardé : {OUTPUT_PATH}")
    print(f"   Résolution : 300 DPI")
    print(f"   Format     : PNG")


if __name__ == "__main__":
    main()
