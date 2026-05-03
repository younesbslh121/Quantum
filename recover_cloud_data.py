#!/usr/bin/env python3
"""
recover_cloud_data.py — Récupération des données cloud depuis IBM Quantum
========================================================================
Auteur  : Boussaiah Younes (INPT - Cloud & IoT)
Date    : Avril 2026
Objectif: Télécharger l'historique des jobs depuis IBM Quantum (puisque 
          le script principal a été interrompu) et générer le graphique.
"""

import os
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from qiskit_ibm_runtime import QiskitRuntimeService

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_PATH = RESULTS_DIR / "recovered_cloud_optimizers.png"

# Style IBM
BG_COLOR = "#0F0F23"
GRID_COLOR = "#2A2A4A"
TEXT_COLOR = "#E8E8F0"
EXACT_ENERGY = -1.8572750302  # Valeur théorique H2

def recover_data():
    print("[CONNECT] Connexion à IBM Quantum pour récupérer vos jobs...")
    token = os.getenv("IBM_QUANTUM_TOKEN")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    
    # On récupère les 35 derniers jobs (30 COBYLA + 5 SPSA)
    print("[DOWNLOAD] Téléchargement de l'historique des jobs (cela peut prendre 1-2 min)...")
    jobs = service.jobs(limit=35)
    
    # Trier par date de création (du plus ancien au plus récent)
    jobs.sort(key=lambda j: j.creation_date)
    
    cobyla_energies = []
    spsa_energies = []
    
    print("[SEARCH] Extraction des énergies...")
    for i, job in enumerate(jobs):
        try:
            status = job.status()
            status_str = status.name if hasattr(status, 'name') else str(status)
            if status_str in ["DONE", "COMPLETED"]:
                evs = job.result()[0].data.evs
                energy = float(np.atleast_1d(evs)[0])
                
                # Les 30 premiers jobs sont COBYLA, les suivants SPSA
                if i < 30:
                    cobyla_energies.append(energy)
                else:
                    spsa_energies.append(energy)
        except Exception as e:
            print(f"  [WARNING] Impossible de lire le job {job.job_id()} : {e}")
            
    return cobyla_energies, spsa_energies

def plot_recovered_data(cobyla_energies, spsa_energies):
    print("[PLOT] Création du graphique...")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Plot COBYLA
    if cobyla_energies:
        x_cobyla = list(range(1, len(cobyla_energies) + 1))
        ax.plot(x_cobyla, cobyla_energies, color="#1192E8", linewidth=2, label="COBYLA (Hardware réel)")
        ax.scatter(x_cobyla, cobyla_energies, color="#1192E8", s=15, alpha=0.5)

    # Plot SPSA
    if spsa_energies:
        x_spsa = list(range(1, len(spsa_energies) + 1))
        ax.plot(x_spsa, spsa_energies, color="#9F1853", linewidth=2, label="SPSA (Hardware réel - Incomplet)")
        ax.scatter(x_spsa, spsa_energies, color="#9F1853", s=15, alpha=0.5)

    # Ligne exacte
    ax.axhline(y=EXACT_ENERGY, color="#DA1E28", linestyle="--", linewidth=1.5, label=f"Énergie exacte ({EXACT_ENERGY} Ha)")

    ax.set_xlabel("Nombre d'évaluations", fontsize=14, fontweight="bold", color=TEXT_COLOR, labelpad=12)
    ax.set_ylabel("Énergie (Hartree)", fontsize=14, fontweight="bold", color=TEXT_COLOR, labelpad=12)
    ax.set_title("VQE H₂ sur vrai processeur IBM — Récupération des données", fontsize=18, fontweight="bold", color=TEXT_COLOR, pad=20)
    
    ax.grid(True, alpha=0.3, color=GRID_COLOR, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    
    ax.legend(loc="upper right", facecolor="#1A1A35", labelcolor=TEXT_COLOR)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[SUCCESS] Graphique généré avec succès : {OUTPUT_PATH}")

def main():
    cobyla, spsa = recover_data()
    print(f"   Données récupérées : {len(cobyla)} points COBYLA, {len(spsa)} points SPSA.")
    if cobyla or spsa:
        plot_recovered_data(cobyla, spsa)
    else:
        print("[ERROR] Aucune donnée terminée trouvée sur votre compte IBM.")

if __name__ == "__main__":
    main()
