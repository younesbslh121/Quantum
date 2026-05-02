#!/usr/bin/env python3
"""
analyze_cloud_optimizers.py — Comparaison COBYLA vs SPSA sur IBM Quantum
========================================================================
Auteur  : Boussaiah Younes (INPT - Cloud & IoT)
Date    : Avril 2026
Objectif: Exécuter le VQE pour H₂ sur un vrai QPU IBM ou simulateur cloud,
          en comparant un optimiseur standard (COBYLA) avec un optimiseur 
          robuste au bruit (SPSA).

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

# Qiskit Algorithms
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA, SPSA

# IBM Runtime — Exécution sur hardware réel
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

H2_ATOM_STRING = "H 0.0 0.0 0.0; H 0.0 0.0 0.735"
H2_BASIS = "sto3g"
H2_CHARGE = 0
H2_SPIN = 0

# Itérations réduites pour économiser le quota sur les vrais QPU
MAX_ITERATIONS = int(os.getenv("VQE_MAX_ITERATIONS", "30"))
BACKEND_NAME = os.getenv("VQE_BACKEND", "")  # Vide = least_busy

RESULTS_DIR = Path(__file__).parent / "results"

# ═══════════════════════════════════════════════════════════════
# Fonctions
# ═══════════════════════════════════════════════════════════════

def connect_ibm_quantum():
    print("\n🔗 Étape 0 : Connexion à IBM Quantum Platform")
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        print("   ❌ IBM_QUANTUM_TOKEN non défini !")
        sys.exit(1)

    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    
    if BACKEND_NAME:
        backend = service.backend(BACKEND_NAME)
        print(f"   🖥️  Backend demandé : {backend.name}")
    else:
        # On peut choisir un simulateur cloud si on veut éviter les queues
        backend = service.least_busy(operational=True, simulator=False)
        print(f"   🖥️  Backend le moins chargé : {backend.name}")

    print(f"   ✅ Connexion établie")
    return service, backend


def define_molecule():
    print("\n🧬 Étape 1 : Définition de H₂")
    driver = PySCFDriver(atom=H2_ATOM_STRING, basis=H2_BASIS, charge=H2_CHARGE, spin=H2_SPIN)
    problem = driver.run()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())

    exact_solver = NumPyMinimumEigensolver()
    exact_energy = exact_solver.compute_minimum_eigenvalue(qubit_op).eigenvalue.real
    print(f"   Énergie exacte (référence) : {exact_energy:.10f} Ha")
    return qubit_op, exact_energy


def prepare_for_hardware(qubit_op, backend):
    print("\n🔧 Étape 2 : Transpilation pour le QPU")
    ansatz = TwoLocal(
        num_qubits=qubit_op.num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        entanglement="linear",
        reps=1,
    )
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_ansatz = pm.run(ansatz)
    isa_observable = qubit_op.apply_layout(isa_ansatz.layout)
    print(f"   ✅ Circuit transpilé pour {backend.name}")
    return isa_ansatz, isa_observable, ansatz.num_parameters


def run_vqe_with_optimizer(optimizer, opt_name, backend, isa_ansatz, isa_observable, num_params):
    print(f"\n🚀 Exécution avec {opt_name}")
    
    iterations = []
    energies = []
    eval_count = [0]

    def cost_function(params):
        eval_count[0] += 1
        job = estimator.run([(isa_ansatz, isa_observable, [params])])
        energy = float(job.result()[0].data.evs)
        iterations.append(eval_count[0])
        energies.append(energy)
        if eval_count[0] % 5 == 0 or eval_count[0] == 1:
            print(f"     [{opt_name}] Éval {eval_count[0]:>2d} | Énergie = {energy:.8f} Ha")
        return energy

    x0 = np.zeros(num_params)
    start_time = time.time()

    with Session(backend=backend) as session:
        global estimator
        estimator = Estimator(session=session)
        result = optimizer.minimize(fun=cost_function, x0=x0)

    elapsed = time.time() - start_time
    print(f"   ✅ {opt_name} terminé en {elapsed:.2f}s | Énergie = {result.fun:.8f} Ha")
    return iterations, energies, result, elapsed


def save_analysis(all_results, exact_energy, backend_name):
    print("\n💾 Sauvegarde des résultats d'analyse")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_path = RESULTS_DIR / "vqe_cloud_optimizers_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "iteration", "energy"])
        for opt_name, data in all_results.items():
            for it, en in zip(data["iterations"], data["energies"]):
                writer.writerow([opt_name, it, f"{en:.10f}"])
    
    # JSON
    summary = {
        "backend": backend_name,
        "exact_energy": exact_energy,
        "optimizers": {}
    }
    for opt_name, data in all_results.items():
        summary["optimizers"][opt_name] = {
            "final_energy": data["result"].fun,
            "elapsed_seconds": data["elapsed"],
            "evaluations": len(data["iterations"]),
            "error_vs_exact": abs(data["result"].fun - exact_energy)
        }
    
    json_path = RESULTS_DIR / "vqe_cloud_optimizers_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"   📄 {csv_path}")
    print(f"   📄 {json_path}")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    🔬 Comparaison COBYLA vs SPSA sur IBM Quantum       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    service, backend = connect_ibm_quantum()
    qubit_op, exact_energy = define_molecule()
    isa_ansatz, isa_observable, num_params = prepare_for_hardware(qubit_op, backend)

    print(f"\n⚠️ Lancement des exécutions (Max itérations = {MAX_ITERATIONS})")
    
    optimizers_to_test = {
        "COBYLA": COBYLA(maxiter=MAX_ITERATIONS),
        "SPSA": SPSA(maxiter=MAX_ITERATIONS)
    }

    all_results = {}
    
    for opt_name, optimizer in optimizers_to_test.items():
        iterations, energies, result, elapsed = run_vqe_with_optimizer(
            optimizer, opt_name, backend, isa_ansatz, isa_observable, num_params
        )
        all_results[opt_name] = {
            "iterations": iterations,
            "energies": energies,
            "result": result,
            "elapsed": elapsed
        }

    save_analysis(all_results, exact_energy, backend.name)
    
    print("\n✅ Analyse terminée. Exécutez le script de tracé pour visualiser.")

if __name__ == "__main__":
    main()
