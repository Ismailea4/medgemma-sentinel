# 🏥 MedGemma Sentinel - Memory & Reporting Module

## Le Cerveau Clinique Offline pour une Santé Universelle

Ce module implémente le composant **"Memory & Reporting Engineer"** de MedGemma Sentinel, un système d'IA médicale autonome fonctionnant 100% hors ligne.

---

## 📋 Architecture du Module

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MedGemma Sentinel - The Scribe                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │  NIGHT   │───▶│   RAP1   │───▶│   DAY    │───▶│   RAP2   │       │
│  │ Surveill.│    │  Report  │    │  Assist. │    │  Report  │       │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘       │
│       │               │               │               │              │
│       └───────────────┴───────────────┴───────────────┘              │
│                              │                                       │
│                    ┌─────────▼─────────┐                            │
│                    │     GraphRAG      │                            │
│                    │  Patient Memory   │                            │
│                    │   (LlamaIndex)    │                            │
│                    └───────────────────┘                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## 📁 Structure du Projet

```
medgemma_project/
├── src/
│   ├── __init__.py
│   ├── orchestration/          # LangGraph State Machine
│   │   ├── __init__.py
│   │   ├── state.py            # État partagé du workflow
│   │   ├── nodes.py            # Nœuds du graphe (Night, Day, Rap1, Rap2)
│   │   └── graph.py            # Construction du graphe LangGraph
│   │
│   ├── memory/                 # GraphRAG avec LlamaIndex
│   │   ├── __init__.py
│   │   ├── patient_graph.py    # Graphe de connaissances patient
│   │   ├── graph_store.py      # Stockage local du graphe
│   │   └── retriever.py        # Récupération contextuelle
│   │
│   ├── reporting/              # Génération de rapports
│   │   ├── __init__.py
│   │   ├── prompts.py          # Prompts MedGemma pour Rap1/Rap2
│   │   ├── templates.py        # Templates de rapports
│   │   └── pdf_generator.py    # Générateur PDF
│   │
│   └── models/                 # Modèles de données
│       ├── __init__.py
│       ├── patient.py          # Modèle Patient
│       ├── vitals.py           # Constantes vitales
│       └── events.py           # Événements cliniques
│
├── data/
│   ├── synthetic/              # Données synthétiques
│   └── reports/                # Rapports générés
│
├── tests/
│   └── test_*.py               # Tests unitaires
│
├── examples/
│   └── demo_workflow.py        # Démonstration complète
│
└── docs/
    └── architecture.md         # Documentation technique
```

---

## 🔧 Composants Principaux

### 1. 🔄 Orchestration LangGraph

Le **State Machine** orchestre le flux de travail clinique:

- **Night Node**: Surveillance nocturne (capteurs, audio, vision IR)
- **Rap1 Node**: Génération du rapport de nuit
- **Day Node**: Assistance médicale spécialisée
- **Rap2 Node**: Génération du rapport de consultation

### 2. 🧠 Memory GraphRAG

Utilise **LlamaIndex** pour créer un graphe de connaissances patient:

- Stockage des antécédents médicaux
- Relations entre symptômes, diagnostics et traitements
- Récupération contextuelle pour enrichir les analyses

### 3. 📄 Reporting System

Génération de rapports cliniques structurés:

- **Rap1**: Rapport de surveillance nocturne
- **Rap2**: Rapport de consultation/triage
- Export PDF avec mise en page professionnelle

---

## 💡 Utilisation

```python
from src.orchestration.graph import MedGemmaSentinelGraph
from src.memory.patient_graph import PatientGraphRAG
from src.models.patient import Patient

# Initialiser le système
sentinel = MedGemmaSentinelGraph()
memory = PatientGraphRAG()

# Créer un patient
patient = Patient(
    id="P001",
    name="Jean Dupont",
    age=67,
    conditions=["Hypertension", "Diabète Type 2"]
)

# Exécuter le workflow complet
result = sentinel.run(patient_id="P001", mode="full_cycle")

# Générer le rapport PDF
result.generate_pdf("rapport_patient_P001.pdf")
```

---

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest tests/ -v

# Test avec données synthétiques
python -m examples.demo_workflow
```

---

## 📊 Données Synthétiques

Le module inclut un générateur de données synthétiques réalistes:

- Constantes vitales (SpO2, FC, Température, PA)
- Événements nocturnes (apnées, agitation, désaturation)
- Consultations médicales (symptômes, examens, diagnostics)

---
