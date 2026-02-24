"""
MedGemma Prompts - Steering prompts for clinical report generation
Implements dynamic steering for Night (Rap1) and Day (Rap2) modes
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from string import Template
from src.reporting.medgemma_engine import MedGemmaEngine


class PromptType(str, Enum):
    """Types of MedGemma prompts"""
    # Night surveillance
    NIGHT_SURVEILLANCE = "night_surveillance"
    NIGHT_ALERT_ANALYSIS = "night_alert_analysis"
    NIGHT_REPORT_GENERATION = "night_report_generation"
    
    # Day consultation
    DAY_CONSULTATION = "day_consultation"
    DAY_DIFFERENTIAL = "day_differential"
    DAY_REPORT_GENERATION = "day_report_generation"
    
    # Specialty modes
    CARDIO_ANALYSIS = "cardio_analysis"
    DERMATO_ANALYSIS = "dermato_analysis"
    OPHTALMO_ANALYSIS = "ophtalmo_analysis"
    
    # General
    TRIAGE_ASSESSMENT = "triage_assessment"
    LONGITUDINAL_ANALYSIS = "longitudinal_analysis"


class SteeringPrompt(BaseModel):
    """
    A steerable prompt for MedGemma.
    Contains the system prompt that "steers" the model's behavior.
    """
    prompt_type: PromptType
    name: str
    description: str
    
    # The core steering content
    system_prompt: str
    user_prompt_template: str
    
    # Output format guidance
    expected_output_format: str = Field(default="text")
    output_sections: List[str] = Field(default_factory=list)
    
    # Constraints
    max_tokens: int = Field(default=-1)
    temperature: float = Field(default=0.3)  # Lower for clinical accuracy
    
    def format_user_prompt(self, **kwargs) -> str:
        """Format the user prompt with provided variables"""
        template = Template(self.user_prompt_template)
        return template.safe_substitute(**kwargs)
    
    def get_full_prompt(self, **kwargs) -> Dict[str, str]:
        """Get the complete prompt structure for LLM"""
        return {
            "system": self.system_prompt,
            "user": self.format_user_prompt(**kwargs)
        }


class MedGemmaPrompts:
    """
    Collection of MedGemma steering prompts for clinical report generation.
    
    Implements the steering mechanism described in the MedGemma Sentinel concept:
    - Night mode: Surveillance, alert analysis
    - Day mode: Consultation assistance, specialty support
    - Report generation: Structured clinical documentation
    """
    
    # ==================== Base System Prompts ====================
    
    BASE_SYSTEM_CONTEXT = """Tu es MedGemma Sentinel, un assistant médical IA embarqué conçu pour fonctionner 100% hors ligne dans des environnements à ressources limitées.

PRINCIPES FONDAMENTAUX:
1. SÉCURITÉ PATIENT: Toujours prioriser la sécurité. En cas de doute, recommander une évaluation médicale.
2. TRANSPARENCE: Indiquer clairement le niveau de confiance et les limitations.
3. AIDE À LA DÉCISION: Tu assistes le personnel soignant, tu ne remplaces pas leur jugement clinique.
4. DOCUMENTATION: Produire des rapports structurés et exploitables.

AVERTISSEMENT LÉGAL:
Ce système est un outil d'aide à la décision. Il ne remplace pas l'examen clinique par un professionnel de santé qualifié. Toute décision thérapeutique doit être validée par un médecin."""

    # ==================== Night Surveillance Prompts ====================
    
    @classmethod
    def get_night_surveillance_prompt(cls) -> SteeringPrompt:
        """Prompt for continuous night surveillance analysis"""
        return SteeringPrompt(
            prompt_type=PromptType.NIGHT_SURVEILLANCE,
            name="Surveillance Nocturne",
            description="Analyse des données de surveillance nocturne multimodales",
            system_prompt=f"""{cls.BASE_SYSTEM_CONTEXT}

MODE: SURVEILLANCE NOCTURNE (Guard Night)

Tu analyses les données de surveillance nocturne en temps réel:
- Capteurs: SpO2, fréquence cardiaque, température
- Audio: Respiration, toux, stridor, plaintes vocales
- Vision IR: Posture, mouvement, chutes

OBJECTIFS:
1. Détecter les anomalies vitales (désaturation, brady/tachycardie, fièvre)
2. Identifier les signes de détresse respiratoire
3. Repérer l'agitation anormale ou les risques de chute
4. Fusionner les signaux multimodaux pour une détection plus fiable

NIVEAUX D'ALERTE:
- CRITIQUE: Intervention immédiate requise (SpO2<85%, FC<40 ou >150, apnée prolongée)
- ÉLEVÉ: Évaluation urgente nécessaire (SpO2 85-90%, anomalies respiratoires)
- MODÉRÉ: Surveillance renforcée recommandée
- FAIBLE: À noter dans le rapport

FUSION MULTIMODALE:
Quand plusieurs sources détectent une anomalie simultanément, augmente le niveau de confiance et la priorité de l'alerte.""",
            user_prompt_template="""DONNÉES DE SURVEILLANCE - Patient: $patient_id

CONTEXTE PATIENT:
$patient_context

DONNÉES CAPTEURS (dernières $time_window minutes):
$vitals_data

ÉVÉNEMENTS AUDIO:
$audio_events

ÉVÉNEMENTS VISION:
$vision_events

Analyse ces données et produis:
1. Liste des anomalies détectées avec niveau d'alerte
2. Corrélations multimodales identifiées
3. Recommandations d'action immédiate si nécessaire""",
            expected_output_format="structured",
            output_sections=["anomalies", "correlations", "recommendations", "alert_level"],
            max_tokens=-1,
            temperature=0.2
        )
    
    @classmethod
    def get_night_report_prompt(cls) -> SteeringPrompt:
        """Prompt for generating Rap1 - Night surveillance report"""
        return SteeringPrompt(
            prompt_type=PromptType.NIGHT_REPORT_GENERATION,
            name="Rapport de Nuit (Rap1)",
            description="Génération du rapport de surveillance nocturne",
            system_prompt=f"""{cls.BASE_SYSTEM_CONTEXT}

MODE: GÉNÉRATION RAPPORT NOCTURNE (Rap1)

Tu génères un rapport structuré de la surveillance nocturne destiné à l'équipe de jour.

STRUCTURE DU RAPPORT:
1. RÉSUMÉ EXÉCUTIF: Vue d'ensemble en 2-3 phrases
2. ALERTES CRITIQUES: Liste des événements graves avec horodatage
3. QUALITÉ DU SOMMEIL: Score et observations
4. ÉVOLUTION DES CONSTANTES: Tendances et anomalies
5. INTERVENTIONS: Actions effectuées pendant la nuit
6. RECOMMANDATIONS: Points d'attention pour l'équipe de jour

STYLE:
- Concis et factuel
- Terminologie médicale appropriée
- Priorisation claire des informations
- Actionnable pour l'équipe de relève""",
            user_prompt_template="""GÉNÉRATION RAPPORT DE NUIT - Patient: $patient_id

IDENTIFICATION:
- Patient: $patient_name
- Chambre: $room
- Date: $date

PÉRIODE: $start_time - $end_time

DONNÉES DE LA NUIT:
$night_summary

ÉVÉNEMENTS DÉTECTÉS:
$events_list

CONSTANTES VITALES (résumé):
$vitals_summary

CONTEXTE MÉDICAL:
$patient_context

Génère le rapport de nuit complet selon la structure définie.""",
            expected_output_format="markdown",
            output_sections=[
                "resume_executif",
                "alertes_critiques", 
                "qualite_sommeil",
                "evolution_constantes",
                "interventions",
                "recommandations"
            ],
            max_tokens=-1,
            temperature=0.3
        )
    
    # ==================== Day Consultation Prompts ====================
    
    @classmethod
    def get_day_consultation_prompt(cls, specialty: str = "general") -> SteeringPrompt:
        """Prompt for day consultation assistance"""
        
        specialty_contexts = {
            "general": """Tu assistes un médecin généraliste. Aide à:
- Structurer l'interrogatoire
- Identifier les signes d'alerte
- Proposer des diagnostics différentiels
- Suggérer des examens complémentaires pertinents""",
            
            "cardio": """MODE SPÉCIALISÉ: CARDIOLOGIE
Tu analyses:
- ECG et rythme cardiaque
- Auscultation cardiaque (souffles, bruits)
- Symptômes cardiovasculaires (douleur thoracique, dyspnée, palpitations)
- Facteurs de risque cardiovasculaire

Évalue le risque cardiovasculaire et propose une stratification.""",
            
            "dermato": """MODE SPÉCIALISÉ: DERMATOLOGIE
Tu analyses les lésions cutanées selon:
- ABCDE: Asymétrie, Bords, Couleur, Diamètre, Évolution
- Morphologie: Macule, papule, nodule, vésicule, etc.
- Distribution et pattern

Propose une classification et les diagnostics à évoquer.""",
            
            "ophtalmo": """MODE SPÉCIALISÉ: OPHTALMOLOGIE
Tu analyses:
- Images de fond d'œil
- Signes visuels rapportés
- Anomalies rétiniennes (hémorragies, exsudats, œdème)

Recherche particulièrement les signes de rétinopathie diabétique, DMLA, glaucome."""
        }
        
        return SteeringPrompt(
            prompt_type=PromptType.DAY_CONSULTATION,
            name=f"Consultation {specialty.capitalize()}",
            description=f"Assistance à la consultation en mode {specialty}",
            system_prompt=f"""{cls.BASE_SYSTEM_CONTEXT}

MODE: ASSISTANCE CONSULTATION JOUR (Medi-Atlas)

{specialty_contexts.get(specialty, specialty_contexts['general'])}

PROCESSUS D'ANALYSE:
1. Recueillir et structurer les symptômes
2. Analyser les données cliniques disponibles
3. Croiser avec le contexte patient (antécédents, traitements)
4. Proposer des diagnostics différentiels HIÉRARCHISÉS
5. Suggérer la conduite à tenir

RAPPEL: Tu proposes, tu n'imposes pas. Le diagnostic final appartient au médecin.""",
            user_prompt_template="""CONSULTATION - Patient: $patient_id
Mode: $specialty

MOTIF DE CONSULTATION:
$presenting_complaint

SYMPTÔMES RAPPORTÉS:
$symptoms

EXAMEN CLINIQUE:
$physical_exam

CONSTANTES:
$vitals

IMAGES/MÉDIAS (si disponibles):
$media_analysis

ANTÉCÉDENTS ET CONTEXTE:
$patient_context

DONNÉES NOCTURNES (si disponibles):
$night_context

Analyse cette consultation et propose:
1. Synthèse clinique
2. Diagnostics différentiels (par ordre de probabilité)
3. Examens complémentaires suggérés
4. Évaluation de la gravité
5. Conduite à tenir proposée""",
            expected_output_format="structured",
            output_sections=[
                "synthese",
                "diagnostics_differentiels",
                "examens_suggeres",
                "evaluation_gravite",
                "conduite_a_tenir"
            ],
            max_tokens=-1,
            temperature=0.3
        )
    
    @classmethod
    def get_day_report_prompt(cls) -> SteeringPrompt:
        """Prompt for generating Rap2 - Day consultation report"""
        return SteeringPrompt(
            prompt_type=PromptType.DAY_REPORT_GENERATION,
            name="Rapport de Consultation (Rap2)",
            description="Génération du rapport de consultation médicale",
            system_prompt=f"""{cls.BASE_SYSTEM_CONTEXT}

MODE: GÉNÉRATION RAPPORT CONSULTATION (Rap2)

Tu génères un rapport de consultation médical structuré.

STRUCTURE DU RAPPORT:
1. IDENTIFICATION: Patient, date, médecin, mode consultation
2. MOTIF: Plainte principale et histoire de la maladie
3. EXAMEN CLINIQUE: Constantes et findings
4. SYNTHÈSE: Analyse des données
5. CONCLUSION: Diagnostic retenu/suspecté
6. PLAN: Traitement et suivi proposés
7. ALERTES: Points de vigilance

FORMAT: Rapport médical formel, exploitable pour le dossier patient.""",
            user_prompt_template="""GÉNÉRATION RAPPORT DE CONSULTATION - Patient: $patient_id

IDENTIFICATION:
- Patient: $patient_name
- Date: $date
- Consultant: $provider
- Mode: $consultation_mode

MOTIF DE CONSULTATION:
$presenting_complaint

ANAMNÈSE:
$symptoms_history

EXAMEN CLINIQUE:
$exam_findings

CONSTANTES VITALES:
$vitals

ANALYSE IA:
- Diagnostics différentiels: $differential_diagnosis
- Gravité évaluée: $severity
- Actions recommandées: $recommendations

CONTEXTE (antécédents):
$patient_context

Génère le rapport de consultation complet.""",
            expected_output_format="markdown",
            output_sections=[
                "identification",
                "motif",
                "anamnese",
                "examen_clinique",
                "synthese",
                "conclusion",
                "plan_traitement",
                "alertes_suivi"
            ],
            max_tokens=-1,
            temperature=0.3
        )
    
    # ==================== Triage & Emergency ====================
    
    @classmethod
    def get_triage_prompt(cls) -> SteeringPrompt:
        """Prompt for emergency triage assessment"""
        return SteeringPrompt(
            prompt_type=PromptType.TRIAGE_ASSESSMENT,
            name="Triage d'Urgence",
            description="Évaluation rapide de la gravité pour triage",
            system_prompt=f"""{cls.BASE_SYSTEM_CONTEXT}

MODE: TRIAGE D'URGENCE

Tu effectues une évaluation rapide pour déterminer le niveau de priorité.

ÉCHELLE DE TRIAGE (inspirée CIMU/ESI):
1. IMMÉDIAT (Rouge): Pronostic vital engagé
2. TRÈS URGENT (Orange): Risque vital potentiel, délai <15min
3. URGENT (Jaune): Soins dans l'heure
4. SEMI-URGENT (Vert): Peut attendre
5. NON URGENT (Bleu): Orientation possible vers soins primaires

CRITÈRES À ÉVALUER:
- Voies aériennes (A)
- Respiration (B)
- Circulation (C)
- Neurologique (D)
- Douleur et symptômes (E)

Sois rapide, précis, et en cas de doute, sur-trier.""",
            user_prompt_template="""TRIAGE - Patient: $patient_id

CONSTANTES:
$vitals

MOTIF D'ARRIVÉE:
$chief_complaint

SYMPTÔMES:
$symptoms

ANTÉCÉDENTS PERTINENTS:
$relevant_history

Effectue le triage et fournis:
1. Niveau de priorité (couleur + numéro)
2. Justification en une phrase
3. Orientation recommandée
4. Actions immédiates si nécessaires""",
            expected_output_format="structured",
            output_sections=["priority_level", "justification", "orientation", "immediate_actions"],
            max_tokens=-1,
            temperature=0.2
        )
    
    # ==================== Longitudinal Analysis ====================
    
    @classmethod
    def get_longitudinal_prompt(cls) -> SteeringPrompt:
        """Prompt for longitudinal trend analysis"""
        return SteeringPrompt(
            prompt_type=PromptType.LONGITUDINAL_ANALYSIS,
            name="Analyse Longitudinale",
            description="Analyse des tendances sur 7-30 jours",
            system_prompt=f"""{cls.BASE_SYSTEM_CONTEXT}

MODE: ANALYSE LONGITUDINALE

Tu analyses l'évolution clinique d'un patient sur plusieurs jours/semaines.

OBJECTIFS:
1. Identifier les tendances (amélioration, stabilité, dégradation)
2. Repérer les patterns récurrents
3. Détecter les dégradations lentes (potentiellement non visibles au jour le jour)
4. Évaluer la réponse aux traitements
5. Anticiper les complications potentielles

MÉTRIQUES À SUIVRE:
- Évolution des constantes vitales
- Fréquence et gravité des alertes
- Qualité du sommeil
- Événements cliniques marquants

Produis une analyse qui aide à la décision clinique à moyen terme.""",
            user_prompt_template="""ANALYSE LONGITUDINALE - Patient: $patient_id
Période: $start_date - $end_date ($days jours)

PROFIL PATIENT:
$patient_context

DONNÉES CHRONOLOGIQUES:
$timeline_data

RÉSUMÉ DES ÉVÉNEMENTS:
$events_summary

TENDANCE DES CONSTANTES:
$vitals_trends

Analyse l'évolution et fournis:
1. Tendance générale (amélioration/stable/dégradation)
2. Patterns identifiés
3. Points d'alerte pour dégradation lente
4. Recommandations pour le suivi""",
            expected_output_format="structured",
            output_sections=["tendance_generale", "patterns", "alertes_degradation", "recommandations_suivi"],
            max_tokens=-1,
            temperature=0.3
        )
    
    # ==================== Factory Method ====================
    
    @classmethod
    def get_prompt(cls, prompt_type: PromptType, **kwargs) -> SteeringPrompt:
        """Factory method to get a prompt by type"""
        prompt_map = {
            PromptType.NIGHT_SURVEILLANCE: cls.get_night_surveillance_prompt,
            PromptType.NIGHT_REPORT_GENERATION: cls.get_night_report_prompt,
            PromptType.DAY_CONSULTATION: lambda: cls.get_day_consultation_prompt(kwargs.get("specialty", "general")),
            PromptType.DAY_REPORT_GENERATION: cls.get_day_report_prompt,
            PromptType.TRIAGE_ASSESSMENT: cls.get_triage_prompt,
            PromptType.LONGITUDINAL_ANALYSIS: cls.get_longitudinal_prompt,
            PromptType.CARDIO_ANALYSIS: lambda: cls.get_day_consultation_prompt("cardio"),
            PromptType.DERMATO_ANALYSIS: lambda: cls.get_day_consultation_prompt("dermato"),
            PromptType.OPHTALMO_ANALYSIS: lambda: cls.get_day_consultation_prompt("ophtalmo"),
        }
        
        getter = prompt_map.get(prompt_type)
        if getter:
            return getter()
        
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    @classmethod
    def list_prompts(cls) -> List[Dict[str, str]]:
        """List all available prompts"""
        return [
            {"type": pt.value, "name": pt.name}
            for pt in PromptType
        ]


class MedGemmaReportGenerator:
    """
    Integrates MedGemma engine with steering prompts
    Generates clinically accurate reports using the quantized model
    """
    
    def __init__(self):
        """Initialize with MedGemma engine"""
        self.engine = MedGemmaEngine()
    
    def generate_night_report(
        self,
        patient_context: Dict[str, Any],
        night_data: Any,
    ) -> str:
        """Generate night report using MedGemma
        
        Args:
            patient_context: Patient info dictionary
            night_data: NightData object or dict with night surveillance data
        """
        patient_str = f"{patient_context.get('name')} (ID: {patient_context.get('id')})"
        
        # Handle NightData object or dict
        if hasattr(night_data, '__dict__') and not isinstance(night_data, dict):
            # It's a Pydantic model or similar object
            night_dict = night_data.__dict__ if hasattr(night_data, '__dict__') else {}
            events = getattr(night_data, 'events', [])
            total_events = len(events) if events else 0
        else:
            # It's a dict
            night_dict = night_data
            events = night_data.get('events', [])
            total_events = night_data.get('total_events', len(events))
        
        night_summary = f"Surveillance period: 21:00-07:00. Events detected: {total_events}"
        
        return self.engine.generate_night_report(
            patient_context=patient_str,
            night_summary=night_summary,
            events=events,
        )
    
    def generate_day_report(
        self,
        patient_context: Dict[str, Any],
        night_context: str,
        day_data: Dict[str, Any],
        specialty: str = "general",
    ) -> str:
        """Generate day report using MedGemma"""
        patient_str = f"{patient_context.get('name')} (ID: {patient_context.get('id')})"
        
        return self.engine.generate_day_report(
            patient_context=patient_str,
            night_context=night_context,
            consultation_data=day_data,
            specialty=specialty,
        )
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get MedGemma engine status"""
        return self.engine.get_status()

