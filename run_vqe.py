#!/usr/bin/env python3
"""
run_vqe.py — Pipeline VQE pour la simulation de H₂
====================================================
Auteur  : Boussaiah Younes (INPT - Cloud & IoT)
Date    : Avril 2026
Objectif: Calculer l'énergie fondamentale de la molécule H₂
          en utilisant l'algorithme VQE sur simulateur local.

Workflow:
  1. Définir la molécule H₂ via PySCFDriver
  2. Mapper l'Hamiltonien fermionique → opérateur qubit (Jordan-Wigner)
  3. Exécuter le VQE avec ansatz TwoLocal + optimiseur COBYLA
  4. Comparer avec la solution exacte (NumPyMinimumEigensolver)
  5. Sauvegarder les résultats (CSV + JSON)
"""

import os
import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

#  Chargement de la clé API (optionnel) 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

#  Imports Qiskit 
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import TwoLocal

# Qiskit Nature - Définition de la molécule
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Qiskit Algorithms - VQE & optimiseurs
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA


# 
# Configuration
# 

# Paramètres de la molécule H₂
H2_ATOM_STRING = "H 0.0 0.0 0.0; H 0.0 0.0 0.735"  # distance en Ångström
H2_BASIS = "sto3g"   # Base minimale STO-3G
H2_CHARGE = 0
H2_SPIN = 0          # Singulet (spin total = 0)

# Paramètres du VQE
MAX_ITERATIONS = 500
OPTIMIZER_NAME = "COBYLA"

# Dossier de sortie
RESULTS_DIR = Path(__file__).parent / "results"


# 
# Callback pour suivre la convergence
# 

class VQEConvergenceTracker:
    """Suit l'évolution de l'énergie à chaque itération du VQE."""

    def __init__(self):
        self.iterations = []
        self.energies = []
        self.eval_count = 0

    def callback(self, eval_count, parameters, mean, std):
        """
        Callback appelé par le VQE à chaque évaluation.

        Args:
            eval_count: Nombre d'évaluations de la fonction coût
            parameters: Paramètres du circuit variationnel
            mean: Valeur moyenne de l'énergie
            std: Écart-type (métadonnées)
        """
        self.eval_count = eval_count
        self.iterations.append(eval_count)
        self.energies.append(float(mean))

        # Affichage en temps réel
        if eval_count % 10 == 0 or eval_count == 1:
            print(f"  ⚡ Itération {eval_count:>4d}  |  Énergie = {mean:>12.8f} Ha")


# 
# Fonctions principales
# 

def define_molecule():
    """Définit la molécule H₂ et retourne le problème électronique."""

    print("\n[INIT] Étape 1 : Définition de la molécule H₂")
    print(f"   Géométrie  : {H2_ATOM_STRING}")
    print(f"   Base        : {H2_BASIS}")
    print(f"   Charge      : {H2_CHARGE}")
    print(f"   Spin        : {H2_SPIN}")

    driver = PySCFDriver(
        atom=H2_ATOM_STRING,
        basis=H2_BASIS,
        charge=H2_CHARGE,
        spin=H2_SPIN,
    )

    problem = driver.run()

    print(f"   [SUCCESS] Problème électronique créé")
    print(f"   Nombre de particules : {problem.num_particles}")
    print(f"   Nombre d'orbitales spatiales : {problem.num_spatial_orbitals}")

    return problem


def map_to_qubit_operator(problem):
    """Mappe l'Hamiltonien fermionique vers un opérateur qubit."""

    print("\n[UPDATE] Étape 2 : Mapping Jordan-Wigner")

    mapper = JordanWignerMapper()

    # Récupérer l'Hamiltonien de seconde quantification
    hamiltonian = problem.hamiltonian.second_q_op()
    print(f"   Opérateur fermionique : {len(hamiltonian)} termes")

    # Mapper vers les qubits
    qubit_op = mapper.map(hamiltonian)
    print(f"   Opérateur qubit (SparsePauliOp) : {qubit_op.num_qubits} qubits, {len(qubit_op)} termes")
    print(f"   [SUCCESS] Mapping terminé")

    return qubit_op, mapper


def compute_exact_energy(qubit_op):
    """Calcule l'énergie exacte via diagonalisation classique."""

    print("\n📐 Étape 3 : Calcul de l'énergie exacte (référence)")

    exact_solver = NumPyMinimumEigensolver()
    exact_result = exact_solver.compute_minimum_eigenvalue(qubit_op)
    exact_energy = exact_result.eigenvalue.real

    print(f"   Énergie exacte = {exact_energy:.10f} Ha")

    return exact_energy


def run_vqe_simulation(qubit_op):
    """Exécute l'algorithme VQE sur simulateur local."""

    print(f"\n[START] Étape 4 : Exécution du VQE")
    print(f"   Optimiseur   : {OPTIMIZER_NAME}")
    print(f"   Max itérations: {MAX_ITERATIONS}")

    # --- Ansatz ---
    ansatz = TwoLocal(
        num_qubits=qubit_op.num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        entanglement="full",
        reps=2,
    )
    print(f"   Ansatz       : TwoLocal (ry, rz + cz)")
    print(f"   Paramètres   : {ansatz.num_parameters}")
    print(f"   Profondeur   : {ansatz.depth()}")

    # --- Estimator (simulateur local) ---
    estimator = StatevectorEstimator()
    print(f"   Backend      : StatevectorEstimator (simulateur local)")

    # --- Optimiseur ---
    optimizer = COBYLA(maxiter=MAX_ITERATIONS)

    # --- Tracker de convergence ---
    tracker = VQEConvergenceTracker()

    # --- Point initial (zéros = proche de l'état HF) ---
    initial_point = np.zeros(ansatz.num_parameters)

    # --- VQE ---
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=tracker.callback,
        initial_point=initial_point,
    )

    print(f"\n   [WAIT] Optimisation en cours...\n")
    start_time = time.time()

    result = vqe.compute_minimum_eigenvalue(qubit_op)

    elapsed = time.time() - start_time

    vqe_energy = result.eigenvalue.real
    print(f"\n   [SUCCESS] VQE terminé en {elapsed:.2f}s")
    print(f"   Énergie VQE = {vqe_energy:.10f} Ha")
    print(f"   Évaluations  = {tracker.eval_count}")

    return result, tracker, elapsed


def save_results(tracker, vqe_result, exact_energy, elapsed_time):
    """Sauvegarde les résultats en CSV et JSON."""

    print(f"\n[SAVE] Étape 5 : Sauvegarde des résultats")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- CSV : convergence iteration par iteration ---
    csv_path = RESULTS_DIR / "vqe_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "energy"])
        for it, en in zip(tracker.iterations, tracker.energies):
            writer.writerow([it, f"{en:.10f}"])

    print(f"   📄 {csv_path}")

    # --- JSON : résumé complet ---
    vqe_energy = vqe_result.eigenvalue.real
    error = abs(vqe_energy - exact_energy)

    summary = {
        "molecule": "H2",
        "geometry": H2_ATOM_STRING,
        "basis": H2_BASIS,
        "timestamp": datetime.now().isoformat(),
        "optimizer": OPTIMIZER_NAME,
        "max_iterations": MAX_ITERATIONS,
        "total_evaluations": tracker.eval_count,
        "elapsed_seconds": round(elapsed_time, 2),
        "vqe_energy_hartree": round(vqe_energy, 10),
        "exact_energy_hartree": round(exact_energy, 10),
        "absolute_error_hartree": round(error, 10),
        "chemical_accuracy_reached": bool(error < 0.0016),  # 1 kcal/mol ≈ 0.0016 Ha
        "optimal_parameters": [round(float(p), 8) for p in vqe_result.optimal_parameters.values()]
            if hasattr(vqe_result.optimal_parameters, 'values')
            else [round(float(p), 8) for p in vqe_result.optimal_parameters],
        "convergence_history": {
            "iterations": tracker.iterations,
            "energies": [round(e, 10) for e in tracker.energies],
        },
    }

    json_path = RESULTS_DIR / "vqe_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"   📄 {json_path}")

    return summary


def print_summary(summary):
    """Affiche un résumé final en console."""

    chem_accuracy = "[SUCCESS] OUI" if summary["chemical_accuracy_reached"] else "[ERROR] NON"

    print("\n")
    print("")
    print("             [SCIENCE] RÉSULTATS VQE — H₂                      ")
    print("")
    print(f"  Énergie VQE    : {summary['vqe_energy_hartree']:>14.10f} Ha          ")
    print(f"  Énergie exacte : {summary['exact_energy_hartree']:>14.10f} Ha          ")
    print(f"  Erreur absolue : {summary['absolute_error_hartree']:>14.10f} Ha          ")
    print(f"  Précision chimique (< 1.6 mHa) : {chem_accuracy}              ")
    print(f"  Évaluations    : {summary['total_evaluations']:>6d}                          ")
    print(f"  Temps          : {summary['elapsed_seconds']:>8.2f} s                        ")
    print("")
    print("  📄 results/vqe_convergence.csv                        ")
    print("  📄 results/vqe_summary.json                           ")
    print("                                                          ")
    print("  ➡️  Lancez `python plot_results.py` pour le graphique  ")
    print("")


# 
# Point d'entrée
# 

def main():
    print("")
    print("       [SCIENCE] VQE — Simulation de la molécule H₂            ")
    print("       Boussaiah Younes · INPT · Cloud & IoT             ")
    print("")

    # Vérification clé API (informative seulement)
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if token and token != "COLLEZ_VOTRE_CLE_API_ICI":
        print(f"\n🔑 Clé API IBM Quantum détectée (***{token[-6:]})")
        print("   ℹ️  Mode simulateur local — la clé n'est pas utilisée")
    else:
        print("\n🔑 Pas de clé API détectée (fichier .env)")
        print("   ℹ️  Ce n'est pas un problème : on utilise le simulateur local")

    # Pipeline
    problem = define_molecule()
    qubit_op, mapper = map_to_qubit_operator(problem)
    exact_energy = compute_exact_energy(qubit_op)
    vqe_result, tracker, elapsed = run_vqe_simulation(qubit_op)
    summary = save_results(tracker, vqe_result, exact_energy, elapsed)
    print_summary(summary)


if __name__ == "__main__":
    main()
