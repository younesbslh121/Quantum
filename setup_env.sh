#!/usr/bin/env bash
# ============================================================
# setup_env.sh — Installation de l'environnement VQE H₂
# Auteur : Boussaiah Younes (INPT - Cloud & IoT)
# ============================================================
set -euo pipefail

VENV_DIR="venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   🔬 VQE H₂ — Installation de l'environnement Python   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── 1. Création du virtualenv ───────────────────────────────
if [ -d "${SCRIPT_DIR}/${VENV_DIR}" ]; then
    echo "⚠️  Le virtualenv '${VENV_DIR}/' existe déjà. Suppression..."
    rm -rf "${SCRIPT_DIR}/${VENV_DIR}"
fi

echo "📦 Création du virtualenv dans ./${VENV_DIR}/ ..."
python3 -m venv "${SCRIPT_DIR}/${VENV_DIR}"

# ─── 2. Activation ──────────────────────────────────────────
echo "🔄 Activation du virtualenv..."
source "${SCRIPT_DIR}/${VENV_DIR}/bin/activate"

# ─── 3. Mise à jour de pip ──────────────────────────────────
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip --quiet

# ─── 4. Installation des dépendances ────────────────────────
echo ""
echo "📥 Installation des paquets Qiskit et dépendances..."
echo "   • qiskit"
echo "   • qiskit-ibm-runtime"
echo "   • qiskit-algorithms"
echo "   • qiskit-nature[pyscf]"
echo "   • matplotlib"
echo "   • python-dotenv"
echo ""

pip install \
    qiskit \
    qiskit-ibm-runtime \
    qiskit-algorithms \
    "qiskit-nature[pyscf]" \
    matplotlib \
    python-dotenv

# ─── 5. Vérification ────────────────────────────────────────
echo ""
echo "✅ Installation terminée ! Vérification des versions :"
echo "─────────────────────────────────────────────────────"
python3 -c "
import qiskit
print(f'  Qiskit          : {qiskit.__version__}')

import qiskit_ibm_runtime
print(f'  IBM Runtime     : {qiskit_ibm_runtime.__version__}')

import qiskit_algorithms
print(f'  Qiskit Algorithms: {qiskit_algorithms.__version__}')

import qiskit_nature
print(f'  Qiskit Nature   : {qiskit_nature.__version__}')

import pyscf
print(f'  PySCF           : {pyscf.__version__}')

import matplotlib
print(f'  Matplotlib      : {matplotlib.__version__}')
"
echo "─────────────────────────────────────────────────────"

# ─── 6. Création du dossier résultats ───────────────────────
mkdir -p "${SCRIPT_DIR}/results"
echo ""
echo "📁 Dossier results/ créé."

# ─── 7. Instructions ────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ✅ Environnement prêt !                               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║   1. Ajoutez votre clé API dans .env :                  ║"
echo "║      IBM_QUANTUM_TOKEN=votre_clé_ici                    ║"
echo "║                                                          ║"
echo "║   2. Activez le virtualenv :                             ║"
echo "║      source venv/bin/activate                            ║"
echo "║                                                          ║"
echo "║   3. Lancez le VQE :                                     ║"
echo "║      python run_vqe.py                                   ║"
echo "║                                                          ║"
echo "║   4. Visualisez les résultats :                          ║"
echo "║      python plot_results.py                              ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
