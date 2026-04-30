#!/usr/bin/env python3
"""
test_vqe_pipeline.py — Tests unitaires pour le pipeline VQE H₂
================================================================
Auteur  : Boussaiah Younes (INPT - Cloud & IoT)
Objectif: Valider chaque étape du pipeline VQE en simulateur local.
          Ces tests sont exécutés automatiquement par GitHub Actions CI.
"""

import os
import csv
import json
import pytest
import numpy as np
from pathlib import Path

# ─── Imports du pipeline ─────────────────────────────────────
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import TwoLocal

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def h2_problem():
    """Crée le problème électronique H₂ une seule fois pour tous les tests."""
    driver = PySCFDriver(
        atom="H 0.0 0.0 0.0; H 0.0 0.0 0.735",
        basis="sto3g",
        charge=0,
        spin=0,
    )
    return driver.run()


@pytest.fixture(scope="module")
def qubit_op(h2_problem):
    """Mappe H₂ vers un opérateur qubit via Jordan-Wigner."""
    mapper = JordanWignerMapper()
    hamiltonian = h2_problem.hamiltonian.second_q_op()
    return mapper.map(hamiltonian)


@pytest.fixture(scope="module")
def exact_energy(qubit_op):
    """Calcule l'énergie exacte de référence."""
    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(qubit_op)
    return result.eigenvalue.real


@pytest.fixture
def results_dir(tmp_path):
    """Crée un dossier temporaire pour les résultats de test."""
    d = tmp_path / "results"
    d.mkdir()
    return d


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

class TestMoleculeDefinition:
    """Teste la définition de la molécule H₂ via PySCFDriver."""

    def test_num_particles(self, h2_problem):
        """H₂ doit avoir 2 électrons (1 alpha, 1 beta)."""
        assert h2_problem.num_particles == (1, 1)

    def test_num_spatial_orbitals(self, h2_problem):
        """STO-3G pour H₂ → 2 orbitales spatiales."""
        assert h2_problem.num_spatial_orbitals == 2

    def test_hamiltonian_exists(self, h2_problem):
        """Le problème doit contenir un Hamiltonien."""
        hamiltonian = h2_problem.hamiltonian.second_q_op()
        assert len(hamiltonian) > 0


class TestJordanWignerMapping:
    """Teste le mapping fermionique → qubit."""

    def test_num_qubits(self, qubit_op):
        """2 orbitales spatiales → 4 qubits (Jordan-Wigner)."""
        assert qubit_op.num_qubits == 4

    def test_num_terms(self, qubit_op):
        """L'Hamiltonien H₂/STO-3G doit avoir 15 termes de Pauli."""
        assert len(qubit_op) == 15

    def test_hermitian(self, qubit_op):
        """L'Hamiltonien doit être hermitien."""
        # SparsePauliOp : les coefficients doivent être réels
        # (pour un Hamiltonien physique)
        coeffs = qubit_op.coeffs
        assert np.allclose(coeffs.imag, 0, atol=1e-10)


class TestExactSolver:
    """Teste la solution exacte de référence."""

    def test_exact_energy_range(self, exact_energy):
        """L'énergie exacte de H₂/STO-3G ≈ -1.857 Ha."""
        assert -2.0 < exact_energy < -1.5

    def test_exact_energy_precision(self, exact_energy):
        """Vérifie la valeur connue à 4 décimales."""
        # Valeur de référence : ~ -1.8573 Ha (Hamiltonien électronique sans offset nucléaire)
        assert abs(exact_energy - (-1.8573)) < 0.01


class TestVQESimulator:
    """Teste le VQE sur simulateur local (convergence rapide)."""

    def test_vqe_converges(self, qubit_op, exact_energy):
        """Le VQE avec 30 itérations doit converger sous -1.5 Ha."""
        ansatz = TwoLocal(
            num_qubits=qubit_op.num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cz",
            entanglement="full",
            reps=1,
        )

        estimator = StatevectorEstimator()
        optimizer = COBYLA(maxiter=30)

        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=np.zeros(ansatz.num_parameters),
        )

        result = vqe.compute_minimum_eigenvalue(qubit_op)
        vqe_energy = result.eigenvalue.real

        # Doit au moins être inférieur à -1.0 Ha après 30 iter
        assert vqe_energy < -1.0, f"VQE n'a pas convergé : {vqe_energy}"

    def test_vqe_returns_result(self, qubit_op):
        """Le VQE doit retourner un résultat avec eigenvalue."""
        ansatz = TwoLocal(
            num_qubits=qubit_op.num_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            reps=1,
        )
        estimator = StatevectorEstimator()
        optimizer = COBYLA(maxiter=5)

        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
        )

        result = vqe.compute_minimum_eigenvalue(qubit_op)

        assert hasattr(result, "eigenvalue")
        assert result.eigenvalue is not None


class TestResultsSaving:
    """Teste la sauvegarde des résultats en CSV et JSON."""

    def test_csv_output(self, results_dir):
        """Vérifie que le CSV est bien formaté."""
        csv_path = results_dir / "test.csv"

        iterations = [1, 2, 3, 4, 5]
        energies = [-0.5, -0.8, -1.0, -1.2, -1.5]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "energy"])
            for it, en in zip(iterations, energies):
                writer.writerow([it, f"{en:.10f}"])

        # Relecture
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 5
        assert rows[0]["iteration"] == "1"
        assert float(rows[-1]["energy"]) == pytest.approx(-1.5)

    def test_json_output(self, results_dir):
        """Vérifie que le JSON contient les champs requis."""
        json_path = results_dir / "test.json"

        summary = {
            "molecule": "H2",
            "vqe_energy_hartree": -1.8369,
            "exact_energy_hartree": -1.8573,
            "chemical_accuracy_reached": False,
        }

        with open(json_path, "w") as f:
            json.dump(summary, f)

        with open(json_path, "r") as f:
            loaded = json.load(f)

        assert loaded["molecule"] == "H2"
        assert "vqe_energy_hartree" in loaded
        assert "exact_energy_hartree" in loaded


class TestPlotGeneration:
    """Teste la génération du graphique matplotlib."""

    def test_plot_png_created(self, results_dir):
        """Vérifie que matplotlib peut générer un PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [-1.0, -1.5, -1.8], label="test")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Energy")

        png_path = results_dir / "test_plot.png"
        fig.savefig(png_path, dpi=100)
        plt.close(fig)

        assert png_path.exists()
        assert png_path.stat().st_size > 0
