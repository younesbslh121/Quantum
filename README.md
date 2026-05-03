# Quantum Chemistry Simulation with Qiskit: VQE for H₂ Molecule

This repository contains a complete pipeline for simulating the ground state energy of the Hydrogen (H₂) molecule using the Variational Quantum Eigensolver (VQE) algorithm. 

It is designed to run both locally via statevector simulation and on real quantum hardware using IBM Quantum's cloud infrastructure.

## 🌟 Key Features

- **Molecular Definition**: Uses PySCF to define the H₂ molecule and extract its quantum mechanics properties (Hamiltonian, mapping to qubits via Parity mapper).
- **Quantum Hardware Integration**: Uses `qiskit-ibm-runtime` (EstimatorV2) to send workloads directly to real IBM Quantum processors (e.g., `ibm_fez`, `ibm_marrakesh`).
- **Optimizer Comparison**: Evaluates and compares classical optimization algorithms suitable for quantum noise:
  - **COBYLA** (Constrained Optimization BY Linear Approximations)
  - **SPSA** (Simultaneous Perturbation Stochastic Approximation) - specifically designed to be robust against quantum hardware noise.
- **Automated Visualization**: Generates high-quality comparative plots of the convergence (energy vs iterations).
- **CI/CD Pipeline**: GitHub Actions workflows for continuous integration (linting) and continuous deployment (quantum cloud execution).

## 🚀 Quick Start

### 1. Environment Setup
A setup script is provided to quickly instantiate an isolated virtual environment and install all dependencies (Qiskit 1.x, Qiskit Nature, IBM Runtime).

```bash
chmod +x setup_env.sh
./setup_env.sh
source venv/bin/activate
```

### 2. Configuration
Create a `.env` file in the root directory to store your IBM Quantum API token:
```text
IBM_QUANTUM_TOKEN="your_token_here"
```

### 3. Execution Modes

**Local Simulation** (Fast, Noiseless, ideal for development):
```bash
python run_vqe.py
python plot_results.py
```

**Cloud Execution** (Real Quantum Hardware):
```bash
python analyze_cloud_optimizers.py
python plot_cloud_optimizers.py
```

## 📊 Results & Analysis
The pipeline automatically tracks the expectation values (energies) at each iteration and saves them into a CSV format. The plotting scripts then generate visual comparisons against the exact theoretical ground state energy of H₂ (-1.857275 Ha).

## 🛠 Technologies
- **Python 3.10+**
- **Qiskit** & **Qiskit Nature** (Quantum algorithm design and molecular chemistry)
- **Qiskit IBM Runtime** (Session management and primitives API)
- **SciPy** (Classical optimizers)
- **Matplotlib** (Data visualization)
- **GitHub Actions** (CI/CD)

---
*Developed by Boussaiah Younes (INPT - Cloud & IoT)*
