#!/usr/bin/env python3
"""
run_vqe_cloud.py — VQE sur processeur quantique IBM réel
==========================================================
Auteur  : Boussaiah Younes (INPT - Cloud & IoT)
Date    : Avril 2026
Objectif: Exécuter le VQE pour H₂ sur un vrai processeur
          quantique IBM via EstimatorV2 + Session.

Requiert: IBM_QUANTUM_TOKEN dans l'environnement ou .env
"""

import os
import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize

# ─── Chargement de la clé API ────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Imports Qiskit ──────────────────────────────────────────
from qiskit.circuit.library import TwoLocal
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit Nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Qiskit Algorithms (solution exacte de référence)
from qiskit_algorithms import NumPyMinimumEigensolver

# IBM Runtime — Exécution sur hardware réel
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

# Molécule H₂
H2_ATOM_STRING = "H 0.0 0.0 0.0; H 0.0 0.0 0.735"
H2_BASIS = "sto3g"
H2_CHARGE = 0
H2_SPIN = 0

# VQE Cloud — paramètres conservateurs pour ménager le quota
MAX_ITERATIONS = int(os.getenv("VQE_MAX_ITERATIONS", "50"))
BACKEND_NAME = os.getenv("VQE_BACKEND", "")  # Vide = least_busy

# Dossier de sortie
RESULTS_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════
# Connexion IBM Quantum
# ═══════════════════════════════════════════════════════════════

def connect_ibm_quantum():
    """Se connecte à IBM Quantum et sélectionne un backend."""

    print("\n🔗 Étape 0 : Connexion à IBM Quantum Platform")

    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        print("   ❌ IBM_QUANTUM_TOKEN non défini !")
        print("   Ajoutez-le dans .env ou en variable d'environnement.")
        sys.exit(1)

    print(f"   🔑 Token détecté (***{token[-6:]})")

    # Connexion au service
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token=token,
    )

    # Sélection du backend
    if BACKEND_NAME:
        backend = service.backend(BACKEND_NAME)
        print(f"   🖥️  Backend demandé : {backend.name}")
    else:
        backend = service.least_busy(operational=True, simulator=False)
        print(f"   🖥️  Backend le moins chargé : {backend.name}")

    print(f"   📊 Qubits : {backend.num_qubits}")
    print(f"   ✅ Connexion établie")

    return service, backend


# ═══════════════════════════════════════════════════════════════
# Définition de la molécule
# ═══════════════════════════════════════════════════════════════

def define_molecule():
    """Définit H₂ et retourne l'opérateur qubit."""

    print("\n🧬 Étape 1 : Définition de la molécule H₂")

    driver = PySCFDriver(
        atom=H2_ATOM_STRING, basis=H2_BASIS,
        charge=H2_CHARGE, spin=H2_SPIN,
    )
    problem = driver.run()

    print(f"   Particules : {problem.num_particles}")
    print(f"   Orbitales spatiales : {problem.num_spatial_orbitals}")

    # Mapping Jordan-Wigner
    mapper = JordanWignerMapper()
    hamiltonian = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(hamiltonian)

    print(f"   Opérateur qubit : {qubit_op.num_qubits} qubits, {len(qubit_op)} termes")

    # Énergie exacte (référence classique)
    exact_solver = NumPyMinimumEigensolver()
    exact_result = exact_solver.compute_minimum_eigenvalue(qubit_op)
    exact_energy = exact_result.eigenvalue.real
    print(f"   Énergie exacte (référence) : {exact_energy:.10f} Ha")

    return qubit_op, exact_energy


# ═══════════════════════════════════════════════════════════════
# Transpilation pour le QPU
# ═══════════════════════════════════════════════════════════════

def prepare_for_hardware(qubit_op, backend):
    """Transpile le circuit ansatz et l'observable pour le QPU cible."""

    print("\n🔧 Étape 2 : Transpilation pour le QPU")

    # Ansatz léger pour hardware (reps=1 = moins profond)
    ansatz = TwoLocal(
        num_qubits=qubit_op.num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        entanglement="linear",  # Linéaire = moins de portes CNOT
        reps=1,
    )

    print(f"   Ansatz : TwoLocal (ry, rz + cz, linear, reps=1)")
    print(f"   Paramètres : {ansatz.num_parameters}")

    # Transpilation optimale pour le backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_ansatz = pm.run(ansatz)

    print(f"   Profondeur transpilée : {isa_ansatz.depth()}")
    print(f"   Portes CX : {isa_ansatz.count_ops().get('cx', 0)}")

    # Adapter l'observable au layout du circuit transpilé
    isa_observable = qubit_op.apply_layout(isa_ansatz.layout)

    print(f"   ✅ Circuit et observable prêts pour {backend.name}")

    return isa_ansatz, isa_observable, ansatz.num_parameters


# ═══════════════════════════════════════════════════════════════
# Boucle VQE sur hardware
# ═══════════════════════════════════════════════════════════════

def run_vqe_on_hardware(backend, isa_ansatz, isa_observable, num_params):
    """Exécute le VQE via EstimatorV2 dans une Session IBM."""

    print(f"\n🚀 Étape 3 : VQE sur {backend.name}")
    print(f"   Optimiseur : COBYLA")
    print(f"   Max itérations : {MAX_ITERATIONS}")

    # Suivi de la convergence
    iterations = []
    energies = []
    eval_count = [0]

    def cost_function(params):
        """Évalue l'énergie sur le QPU via EstimatorV2."""
        eval_count[0] += 1
        job = estimator.run([(isa_ansatz, isa_observable, [params])])
        result = job.result()[0]
        energy = float(result.data.evs)

        iterations.append(eval_count[0])
        energies.append(energy)

        if eval_count[0] % 5 == 0 or eval_count[0] == 1:
            print(f"  ⚡ Éval {eval_count[0]:>4d}  |  Énergie = {energy:>12.8f} Ha")

        return energy

    # Point initial
    x0 = np.zeros(num_params)

    print(f"\n   ⏳ Optimisation en cours sur le QPU...\n")
    start_time = time.time()

    # Session IBM : maintient la priorité entre les itérations
    with Session(backend=backend) as session:
        estimator = Estimator(session=session)

        result = minimize(
            cost_function,
            x0=x0,
            method="COBYLA",
            options={"maxiter": MAX_ITERATIONS, "disp": False},
        )

    elapsed = time.time() - start_time

    print(f"\n   ✅ VQE terminé en {elapsed:.2f}s")
    print(f"   Énergie finale = {result.fun:.10f} Ha")
    print(f"   Évaluations = {eval_count[0]}")

    return result, iterations, energies, elapsed


# ═══════════════════════════════════════════════════════════════
# Sauvegarde des résultats
# ═══════════════════════════════════════════════════════════════

def save_results(iterations, energies, opt_result, exact_energy, elapsed, backend_name):
    """Sauvegarde les résultats cloud en CSV et JSON."""

    print(f"\n💾 Étape 4 : Sauvegarde des résultats")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = RESULTS_DIR / "vqe_cloud_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "energy"])
        for it, en in zip(iterations, energies):
            writer.writerow([it, f"{en:.10f}"])
    print(f"   📄 {csv_path}")

    # JSON
    vqe_energy = float(opt_result.fun)
    error = abs(vqe_energy - exact_energy)

    summary = {
        "molecule": "H2",
        "geometry": H2_ATOM_STRING,
        "basis": H2_BASIS,
        "backend": backend_name,
        "execution_mode": "real_quantum_hardware",
        "timestamp": datetime.now().isoformat(),
        "optimizer": "COBYLA",
        "max_iterations": MAX_ITERATIONS,
        "total_evaluations": len(iterations),
        "elapsed_seconds": round(elapsed, 2),
        "vqe_energy_hartree": round(vqe_energy, 10),
        "exact_energy_hartree": round(exact_energy, 10),
        "absolute_error_hartree": round(error, 10),
        "chemical_accuracy_reached": bool(error < 0.0016),
        "optimal_parameters": [round(float(p), 8) for p in opt_result.x],
        "convergence_history": {
            "iterations": iterations,
            "energies": [round(e, 10) for e in energies],
        },
    }

    json_path = RESULTS_DIR / "vqe_cloud_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"   📄 {json_path}")

    return summary


# ═══════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    🔬 VQE H₂ — Exécution sur processeur quantique IBM  ║")
    print("║    Boussaiah Younes · INPT · Cloud & IoT                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Connexion
    service, backend = connect_ibm_quantum()

    # Molécule
    qubit_op, exact_energy = define_molecule()

    # Transpilation
    isa_ansatz, isa_observable, num_params = prepare_for_hardware(qubit_op, backend)

    # VQE sur QPU
    opt_result, iterations, energies, elapsed = run_vqe_on_hardware(
        backend, isa_ansatz, isa_observable, num_params
    )

    # Sauvegarde
    summary = save_results(
        iterations, energies, opt_result,
        exact_energy, elapsed, backend.name
    )

    # Résumé
    chem = "✅ OUI" if summary["chemical_accuracy_reached"] else "❌ NON"
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          🔬 RÉSULTATS VQE CLOUD — H₂                   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Backend        : {summary['backend']:<28s}         ║")
    print(f"║  Énergie VQE    : {summary['vqe_energy_hartree']:>14.10f} Ha          ║")
    print(f"║  Énergie exacte : {summary['exact_energy_hartree']:>14.10f} Ha          ║")
    print(f"║  Erreur absolue : {summary['absolute_error_hartree']:>14.10f} Ha          ║")
    print(f"║  Précision chimique : {chem}                           ║")
    print(f"║  Évaluations    : {summary['total_evaluations']:>6d}                          ║")
    print(f"║  Temps          : {summary['elapsed_seconds']:>8.2f} s                        ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
