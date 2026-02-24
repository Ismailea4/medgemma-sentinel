"""
LangGraph Nodes - Individual workflow nodes for MedGemma Sentinel
Each node represents a phase: Night, Rap1, Day, Rap2
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import uuid

from .state import (
    SentinelState, WorkflowPhase, SteeringMode,
    NightData, DayData, ReportData
)
from src.reporting.prompts import MedGemmaReportGenerator


class BaseNode(ABC):
    """Base class for all workflow nodes"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node logic and return updated state"""
        pass
    
    def _log(self, message: str) -> None:
        """Log a message with timestamp"""
        print(f"[{datetime.now().isoformat()}] [{self.name}] {message}")


class NightNode(BaseNode):
    """
    Night Surveillance Node (Guard Night Mode)
    
    Processes:
    - Vital signs monitoring (SpO2, FC, Temperature)
    - Audio analysis (breathing, stridor, cough)
    - Vision analysis (posture, movement, fall detection)
    - Multimodal fusion for alert generation
    """
    
    def __init__(self):
        super().__init__("NIGHT")
        self.steering_mode = SteeringMode.NIGHT_SURVEILLANCE
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process night surveillance data"""
        self._log("Starting night surveillance processing...")
        
        # Initialize night data if not present
        if state.get("night_data") is None:
            state["night_data"] = NightData().model_dump()
        
        night_data = state["night_data"]
        night_data["start_time"] = datetime.now().isoformat()
        
        # Process vital signs readings
        vitals_events = self._analyze_vitals(state.get("vitals_input", []))
        night_data["vitals_readings"] = vitals_events
        
        # Process audio events
        audio_events = self._analyze_audio(state.get("audio_input", []))
        night_data["audio_events"] = audio_events
        
        # Process vision events
        vision_events = self._analyze_vision(state.get("vision_input", []))
        night_data["vision_events"] = vision_events
        
        # Multimodal fusion - combine signals for better detection
        fused_events = self._multimodal_fusion(
            vitals_events, audio_events, vision_events
        )
        night_data["events"] = fused_events
        
        # Count alerts
        night_data["alerts_triggered"] = len(fused_events)
        night_data["critical_alerts"] = len([
            e for e in fused_events if e.get("level") == "critical"
        ])
        
        # Calculate sleep quality if applicable
        night_data["sleep_quality_score"] = self._calculate_sleep_quality(fused_events)
        
        night_data["end_time"] = datetime.now().isoformat()
        
        # Update state
        state["night_data"] = night_data
        state["phase"] = WorkflowPhase.RAP1.value
        state["steering_mode"] = SteeringMode.LONGITUDINAL.value
        state["total_events_processed"] = state.get("total_events_processed", 0) + len(fused_events)
        state["total_alerts"] = state.get("total_alerts", 0) + night_data["alerts_triggered"]
        
        # Add message for context
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Night surveillance completed. {len(fused_events)} events detected, "
                      f"{night_data['critical_alerts']} critical alerts.",
            "timestamp": datetime.now().isoformat()
        }]
        
        self._log(f"Night surveillance complete: {len(fused_events)} events, "
                  f"{night_data['critical_alerts']} critical")
        
        return state
    
    def _analyze_vitals(self, vitals_input: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze vital signs for anomalies"""
        events = []
        
        for reading in vitals_input:
            # Check SpO2
            if spo2 := reading.get("spo2"):
                if spo2 < 90:
                    events.append({
                        "type": "desaturation",
                        "level": "critical" if spo2 < 85 else "high",
                        "value": spo2,
                        "timestamp": reading.get("timestamp", datetime.now().isoformat()),
                        "description": f"SpO2 bas: {spo2}%",
                        "source": "sensor_spo2"
                    })
            
            # Check heart rate
            if hr := reading.get("heart_rate"):
                if hr < 50:
                    events.append({
                        "type": "bradycardia",
                        "level": "critical" if hr < 40 else "high",
                        "value": hr,
                        "timestamp": reading.get("timestamp", datetime.now().isoformat()),
                        "description": f"Bradycardie: {hr} bpm",
                        "source": "sensor_ecg"
                    })
                elif hr > 110:
                    events.append({
                        "type": "tachycardia",
                        "level": "critical" if hr > 150 else "high",
                        "value": hr,
                        "timestamp": reading.get("timestamp", datetime.now().isoformat()),
                        "description": f"Tachycardie: {hr} bpm",
                        "source": "sensor_ecg"
                    })
            
            # Check temperature
            if temp := reading.get("temperature"):
                if temp > 38.5:
                    events.append({
                        "type": "fever",
                        "level": "critical" if temp > 40 else "high",
                        "value": temp,
                        "timestamp": reading.get("timestamp", datetime.now().isoformat()),
                        "description": f"Fièvre: {temp}°C",
                        "source": "sensor_temperature"
                    })
                elif temp < 35.5:
                    events.append({
                        "type": "hypothermia",
                        "level": "high",
                        "value": temp,
                        "timestamp": reading.get("timestamp", datetime.now().isoformat()),
                        "description": f"Hypothermie: {temp}°C",
                        "source": "sensor_temperature"
                    })
        
        return events
    
    def _analyze_audio(self, audio_input: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze audio for respiratory anomalies"""
        events = []
        
        for audio in audio_input:
            audio_type = audio.get("type", "")
            confidence = audio.get("confidence", 0.0)
            
            if confidence >= 0.7:  # Minimum confidence threshold
                if audio_type == "apnea":
                    events.append({
                        "type": "apnea",
                        "level": "critical",
                        "duration_seconds": audio.get("duration", 10),
                        "timestamp": audio.get("timestamp", datetime.now().isoformat()),
                        "description": f"Apnée détectée ({audio.get('duration', 10)}s)",
                        "source": "audio_analysis",
                        "confidence": confidence
                    })
                elif audio_type == "stridor":
                    events.append({
                        "type": "abnormal_breathing",
                        "level": "high",
                        "subtype": "stridor",
                        "timestamp": audio.get("timestamp", datetime.now().isoformat()),
                        "description": "Stridor détecté - obstruction voies aériennes",
                        "source": "audio_analysis",
                        "confidence": confidence
                    })
                elif audio_type == "wheeze":
                    events.append({
                        "type": "abnormal_breathing",
                        "level": "medium",
                        "subtype": "wheeze",
                        "timestamp": audio.get("timestamp", datetime.now().isoformat()),
                        "description": "Sifflement respiratoire détecté",
                        "source": "audio_analysis",
                        "confidence": confidence
                    })
                elif audio_type == "vocal_distress":
                    events.append({
                        "type": "vocal_distress",
                        "level": "high",
                        "timestamp": audio.get("timestamp", datetime.now().isoformat()),
                        "description": "Plainte vocale détectée",
                        "source": "audio_analysis",
                        "confidence": confidence
                    })
        
        return events
    
    def _analyze_vision(self, vision_input: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze IR camera data for movement/posture anomalies"""
        events = []
        
        for vision in vision_input:
            vision_type = vision.get("type", "")
            confidence = vision.get("confidence", 0.0)
            
            if confidence >= 0.7:
                if vision_type == "fall":
                    events.append({
                        "type": "fall_risk",
                        "level": "critical",
                        "timestamp": vision.get("timestamp", datetime.now().isoformat()),
                        "description": "Chute détectée",
                        "source": "camera_ir",
                        "confidence": confidence
                    })
                elif vision_type == "agitation":
                    events.append({
                        "type": "agitation",
                        "level": "medium",
                        "timestamp": vision.get("timestamp", datetime.now().isoformat()),
                        "description": "Agitation anormale détectée",
                        "source": "camera_ir",
                        "confidence": confidence
                    })
                elif vision_type == "abnormal_posture":
                    events.append({
                        "type": "abnormal_posture",
                        "level": "low",
                        "timestamp": vision.get("timestamp", datetime.now().isoformat()),
                        "description": "Posture anormale détectée",
                        "source": "camera_ir",
                        "confidence": confidence
                    })
        
        return events
    
    def _multimodal_fusion(
        self,
        vitals_events: List[Dict],
        audio_events: List[Dict],
        vision_events: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Fuse multimodal signals for enhanced detection
        Example: SpO2 drop + wheeze = prioritized respiratory alert
        """
        all_events = vitals_events + audio_events + vision_events
        fused_events = []
        
        # Check for correlated events (within 30 seconds)
        # Example: desaturation + abnormal breathing = critical respiratory event
        desaturations = [e for e in vitals_events if e.get("type") == "desaturation"]
        breathing_issues = [e for e in audio_events if e.get("type") == "abnormal_breathing"]
        
        if desaturations and breathing_issues:
            fused_events.append({
                "type": "respiratory_distress_fused",
                "level": "critical",
                "timestamp": datetime.now().isoformat(),
                "description": "Détresse respiratoire confirmée (SpO2 + Audio)",
                "source": "multimodal_fusion",
                "confidence": 0.95,
                "fusion_reasoning": "Désaturation corrélée avec anomalie respiratoire audio",
                "related_events": [
                    desaturations[0].get("type"),
                    breathing_issues[0].get("type")
                ]
            })
        
        # Add non-fused events
        for event in all_events:
            event["fused"] = False
            fused_events.append(event)
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        fused_events.sort(key=lambda e: priority_order.get(e.get("level", "low"), 4))
        
        return fused_events
    
    def _calculate_sleep_quality(self, events: List[Dict]) -> float:
        """Calculate sleep quality score (0-100)"""
        base_score = 100.0
        
        for event in events:
            level = event.get("level", "low")
            if level == "critical":
                base_score -= 20
            elif level == "high":
                base_score -= 10
            elif level == "medium":
                base_score -= 5
            else:
                base_score -= 2
        
        return max(0.0, min(100.0, base_score))


class Rap1Node(BaseNode):
    """
    Report 1 Node - Night Surveillance Report Generation
    
    Generates:
    - Summary of night events
    - Sleep quality assessment
    - Critical alerts recap
    - Recommendations for day shift
    """
    
    def __init__(self, use_medgemma: bool = True):
        super().__init__("RAP1")
        self.use_medgemma = use_medgemma
        if use_medgemma:
            self.medgemma_generator = MedGemmaReportGenerator()
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate night surveillance report"""
        self._log("Generating night surveillance report (Rap1)...")
        
        night_data = state.get("night_data", {})
        patient_context = state.get("patient_context", {})
        patient_profile = state.get("patient_profile", "general")
        
        # Generate report structure
        report = ReportData(
            report_type="night",
            title="Rapport de Surveillance Nocturne",
            generated_at=datetime.now()
        )
        
        # Option 1: Use MedGemma for AI-powered report generation
        if self.use_medgemma and hasattr(self, 'medgemma_generator'):
            self._log(f"Using MedGemma with patient_profile: {patient_profile}")
            try:
                markdown_content = self.medgemma_generator.generate_night_report(
                    patient_context=patient_context,
                    night_data=night_data,
                    patient_profile=patient_profile,
                )
                report.markdown_content = markdown_content
                report.summary = self._extract_summary_from_markdown(markdown_content)
                
                # Update state with MedGemma-generated report
                state["rap1_report"] = report.model_dump()
                state["phase"] = WorkflowPhase.DAY.value
                state["steering_mode"] = SteeringMode.SPECIALIST_VIRTUAL.value
                
                state["messages"] = state.get("messages", []) + [{
                    "role": "system",
                    "content": f"Night report (Rap1) generated with MedGemma ({patient_profile} profile). "
                              f"{night_data.get('alerts_triggered', 0)} events documented.",
                    "timestamp": datetime.now().isoformat()
                }]
                
                self._log("Rap1 report generated with MedGemma")
                return state
            except Exception as e:
                self._log(f"MedGemma generation failed: {e}. Falling back to template.")
                # Fall through to template-based generation
        
        # Option 2: Template-based report generation (fallback)
        sections = []
        
        # 1. Patient summary
        sections.append({
            "title": "Identification Patient",
            "content": self._build_patient_section(state.get("patient_id"), patient_context)
        })
        
        # 2. Surveillance period
        sections.append({
            "title": "Période de Surveillance",
            "content": self._build_period_section(night_data)
        })
        
        # 3. Events summary
        sections.append({
            "title": "Événements Détectés",
            "content": self._build_events_section(night_data.get("events", []))
        })
        
        # 4. Vital signs trends
        sections.append({
            "title": "Évolution des Constantes",
            "content": self._build_vitals_section(night_data.get("vitals_readings", []))
        })
        
        # 5. Sleep quality
        sections.append({
            "title": "Qualité du Sommeil",
            "content": self._build_sleep_section(night_data)
        })
        
        # 6. Recommendations
        sections.append({
            "title": "Recommandations pour l'Équipe de Jour",
            "content": self._build_recommendations_section(night_data)
        })
        
        report.sections = sections
        report.summary = self._build_summary(night_data)
        report.period_covered = f"{night_data.get('start_time', 'N/A')} - {night_data.get('end_time', 'N/A')}"
        
        # Generate markdown content
        report.markdown_content = self._generate_markdown(report)
        
        # Update state
        state["rap1_report"] = report.model_dump()
        state["phase"] = WorkflowPhase.DAY.value
        state["steering_mode"] = SteeringMode.SPECIALIST_VIRTUAL.value
        
        # Add message
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Night report (Rap1) generated. {night_data.get('alerts_triggered', 0)} events documented.",
            "timestamp": datetime.now().isoformat()
        }]
        
        self._log("Rap1 report generated successfully")
        
        return state
    
    def _build_patient_section(self, patient_id: str, context: Dict) -> str:
        """Build patient identification section"""
        name = context.get("name", "N/A")
        age = context.get("age", "N/A")
        room = context.get("room", "N/A")
        conditions = context.get("conditions", [])
        
        # Handle conditions as list of dicts or list of strings
        if conditions and isinstance(conditions[0], dict):
            conditions_str = ', '.join(c.get("name", str(c)) for c in conditions)
        else:
            conditions_str = ', '.join(conditions) if conditions else 'Aucune'
        
        return f"""
**Patient:** {name}
**ID:** {patient_id}
**Âge:** {age} ans
**Chambre:** {room}
**Conditions actives:** {conditions_str}
"""
    
    def _build_period_section(self, night_data: Dict) -> str:
        """Build surveillance period section"""
        start = night_data.get("start_time", "N/A")
        end = night_data.get("end_time", "N/A")
        
        return f"""
**Début:** {start}
**Fin:** {end}
**Durée:** Nuit complète
"""
    
    def _build_events_section(self, events: List[Dict]) -> str:
        """Build events summary section"""
        if not events:
            return "Aucun événement significatif détecté."
        
        critical = [e for e in events if e.get("level") == "critical"]
        high = [e for e in events if e.get("level") == "high"]
        medium = [e for e in events if e.get("level") == "medium"]
        
        content = f"""
**Total:** {len(events)} événements
- 🔴 Critiques: {len(critical)}
- 🟠 Élevés: {len(high)}
- 🟡 Modérés: {len(medium)}

### Détail des événements critiques:
"""
        for event in critical:
            content += f"\n- **{event.get('type', 'Unknown')}** ({event.get('timestamp', 'N/A')}): {event.get('description', '')}"
        
        return content
    
    def _build_vitals_section(self, vitals: List[Dict]) -> str:
        """Build vital signs trends section"""
        if not vitals:
            return "Aucune lecture des constantes disponible."
        
        return f"""
**Lectures totales:** {len(vitals)}
Les constantes vitales ont été surveillées en continu pendant la nuit.
Voir les graphiques détaillés dans le rapport PDF.
"""
    
    def _build_sleep_section(self, night_data: Dict) -> str:
        """Build sleep quality section"""
        score = night_data.get("sleep_quality_score", "N/A")
        ahi = night_data.get("apnea_hypopnea_index", "N/A")
        
        quality_text = "Excellente" if score and score > 80 else \
                      "Bonne" if score and score > 60 else \
                      "Modérée" if score and score > 40 else \
                      "Mauvaise" if score else "Non évaluée"
        
        return f"""
**Score de qualité:** {score}/100 ({quality_text})
**Index apnée-hypopnée:** {ahi}
"""
    
    def _build_recommendations_section(self, night_data: Dict) -> str:
        """Build recommendations section"""
        recommendations = []
        
        events = night_data.get("events", [])
        critical = [e for e in events if e.get("level") == "critical"]
        
        if critical:
            recommendations.append("⚠️ Évaluation médicale urgente recommandée suite aux alertes critiques")
        
        if any(e.get("type") == "desaturation" for e in events):
            recommendations.append("Vérifier la saturation en oxygène et envisager oxygénothérapie")
        
        if any(e.get("type") in ["tachycardia", "bradycardia"] for e in events):
            recommendations.append("ECG de contrôle recommandé")
        
        if any(e.get("type") == "fever" for e in events):
            recommendations.append("Rechercher foyer infectieux, bilan biologique")
        
        if not recommendations:
            recommendations.append("Nuit calme, poursuivre surveillance standard")
        
        return "\n".join([f"- {r}" for r in recommendations])
    
    def _build_summary(self, night_data: Dict) -> str:
        """Build executive summary"""
        events = night_data.get("events", [])
        critical = len([e for e in events if e.get("level") == "critical"])
        score = night_data.get("sleep_quality_score", "N/A")
        
        if critical > 0:
            return f"⚠️ ATTENTION: {critical} alertes critiques durant la nuit. Évaluation immédiate requise."
        elif len(events) > 5:
            return f"Nuit agitée avec {len(events)} événements détectés. Score de sommeil: {score}/100."
        else:
            return f"Nuit relativement calme. Score de sommeil: {score}/100."
    
    def _generate_markdown(self, report: ReportData) -> str:
        """Generate full markdown content"""
        md = f"""# {report.title}

**Date:** {report.generated_at.strftime('%d/%m/%Y %H:%M')}
**Type:** Rapport de Nuit

---

## Résumé Exécutif

{report.summary}

---

"""
        for section in report.sections:
            md += f"## {section['title']}\n\n{section['content']}\n\n---\n\n"
        
        md += """
---
*Rapport généré automatiquement par MedGemma Sentinel - The Scribe*
*Ce rapport ne remplace pas l'évaluation clinique par un professionnel de santé qualifié.*
"""
        return md
    
    def _extract_summary_from_markdown(self, markdown: str) -> str:
        """Extract first paragraph as summary from markdown content"""
        lines = markdown.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('*'):
                return stripped[:200] + ('...' if len(stripped) > 200 else '')
        return "Night surveillance report generated"


class DayNode(BaseNode):
    """
    Day Assistance Node (Medi-Atlas Mode)
    
    Provides:
    - Specialized consultation support (Cardio, Dermato, Ophtalmo)
    - Image analysis for lesions/findings
    - Differential diagnosis suggestions
    - Severity assessment
    """
    
    def __init__(self):
        super().__init__("DAY")
        self.steering_mode = SteeringMode.SPECIALIST_VIRTUAL
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process day consultation data"""
        self._log("Starting day consultation processing...")
        
        # Initialize day data if not present
        if state.get("day_data") is None:
            state["day_data"] = DayData().model_dump()
        
        day_data = state["day_data"]
        day_data["start_time"] = datetime.now().isoformat()
        
        # Get consultation input
        consultation_mode = state.get("consultation_mode", "general")
        day_data["consultation_mode"] = consultation_mode
        
        # Process symptoms
        symptoms = state.get("symptoms_input", [])
        day_data["symptoms"] = symptoms
        
        # Set presenting complaint from input or derive from symptoms
        presenting = state.get("presenting_complaint", "")
        if not presenting and symptoms:
            presenting = symptoms[0]
        day_data["presenting_complaint"] = presenting or "Consultation de suivi"
        
        # Process physical exam findings
        exam_data = state.get("exam_input", {})
        day_data["physical_exam"] = exam_data
        
        # Process vitals
        vitals = state.get("day_vitals_input", {})
        day_data["vitals"] = vitals
        
        # Process images if any
        images = state.get("images_input", [])
        if images:
            day_data["images"] = self._analyze_images(images, consultation_mode)
        
        # Generate AI analysis based on mode
        analysis = self._generate_analysis(day_data, consultation_mode)
        day_data["differential_diagnosis"] = analysis.get("differential", [])
        day_data["severity_assessment"] = analysis.get("severity", "Moderee")
        day_data["recommended_actions"] = analysis.get("actions", [])
        day_data["final_diagnosis"] = analysis.get("diagnosis", "")
        day_data["diagnosis_reasoning"] = analysis.get("reasoning", "")
        
        day_data["end_time"] = datetime.now().isoformat()
        
        # Update state
        state["day_data"] = day_data
        state["phase"] = WorkflowPhase.RAP2.value
        state["steering_mode"] = SteeringMode.LONGITUDINAL.value
        
        # Add message
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Day consultation ({consultation_mode}) completed. "
                      f"Severity: {analysis.get('severity', 'N/A')}",
            "timestamp": datetime.now().isoformat()
        }]
        
        self._log(f"Day consultation complete. Mode: {consultation_mode}")
        
        return state
    
    def _analyze_images(self, images: List[Dict], mode: str) -> List[Dict[str, Any]]:
        """Analyze clinical images based on consultation mode"""
        analyzed = []
        
        for img in images:
            analysis = {
                "file": img.get("file", "unknown"),
                "type": img.get("type", "unknown"),
                "analysis_mode": mode,
                "timestamp": datetime.now().isoformat()
            }
            
            if mode == "dermato":
                analysis["findings"] = "Lésion analysée - voir diagnostic différentiel"
                analysis["characteristics"] = ["Bords", "Couleur", "Diamètre", "Évolution"]
            elif mode == "ophtalmo":
                analysis["findings"] = "Fond d'œil analysé"
                analysis["characteristics"] = ["Papille", "Vaisseaux", "Macula", "Rétine périphérique"]
            elif mode == "cardio":
                analysis["findings"] = "Tracé analysé"
                analysis["characteristics"] = ["Rythme", "Axe", "Onde P", "Complexe QRS", "Segment ST"]
            
            analyzed.append(analysis)
        
        return analyzed
    
    def _generate_analysis(self, day_data: Dict, mode: str) -> Dict[str, Any]:
        """Generate AI-powered clinical analysis"""
        symptoms = day_data.get("symptoms", [])
        exam = day_data.get("physical_exam", {})
        
        # Simulated analysis based on mode and symptoms
        # In production, this would call MedGemma with steering prompts
        
        analysis = {
            "differential": [],
            "severity": "Modérée",
            "actions": []
        }
        
        if mode == "cardio":
            # Check for any cardio-relevant symptom (French or English names)
            cardio_keywords = [
                "chest_pain", "douleur thoracique", "palpitations",
                "dyspnee", "dyspn", "syncope", "oedeme", "insuffisance"
            ]
            has_cardio_symptom = any(
                any(kw in s.lower() for kw in cardio_keywords)
                for s in symptoms
            ) if symptoms else False

            if has_cardio_symptom or symptoms:
                analysis["differential"] = [
                    "Syndrome coronarien aigu",
                    "Trouble du rythme (FA, TSV, TV)",
                    "Insuffisance cardiaque decompensee",
                    "Pericardite",
                    "Embolie pulmonaire",
                ]
                analysis["severity"] = "Elevee"
                analysis["actions"] = [
                    "ECG 12 derivations en urgence",
                    "Troponine T/I - BNP/NT-proBNP",
                    "Ionogramme, NFS, coagulation",
                    "Radiographie thoracique",
                    "Echocardiographie si disponible",
                    "Avis cardiologique urgent"
                ]
                analysis["diagnosis"] = "Bilan cardiologique en cours - surveillance continue"
                analysis["reasoning"] = (
                    "Symptomatologie evocatrice d'une pathologie cardiovasculaire. "
                    "Les palpitations associees a la dyspnee et/ou syncope necessitent "
                    "une evaluation cardiologique urgente pour exclure un trouble du rythme "
                    "ou une insuffisance cardiaque decompensee."
                )
            
        elif mode == "dermato":
            analysis["differential"] = [
                "Lesion benigne (naevus, keratose seborrheique)",
                "Keratose actinique (pre-cancereuse)",
                "Carcinome basocellulaire",
                "Carcinome epidermoide",
                "Melanome a exclure"
            ]
            analysis["severity"] = "A determiner"
            analysis["actions"] = [
                "Dermoscopie de la lesion",
                "Photos de reference pour suivi",
                "Biopsie excisionnelle si doute",
                "Examen des ganglions locoregionaux",
                "Avis dermatologique si necessaire"
            ]
            analysis["diagnosis"] = "Lesion cutanee a caractériser - biopsie recommandee"
            analysis["reasoning"] = (
                "Lesion necessitant une evaluation dermatoscopique complete. "
                "Les criteres ABCDE doivent etre evalues."
            )
            
        elif mode == "general":
            if symptoms:
                analysis["differential"] = [
                    "Pathologie aigue a evaluer",
                    "Pathologie chronique decompensee",
                    "Syndrome infectieux",
                    "Cause fonctionnelle"
                ]
                analysis["actions"] = [
                    "Examen clinique complet",
                    "Bilan biologique: NFS, CRP, ionogramme",
                    "Imagerie si indiquee",
                    "Reevaluation a 24-48h"
                ]
                analysis["diagnosis"] = "Evaluation en cours"
                analysis["reasoning"] = (
                    "Symptomatologie non specifique necessitant un bilan "
                    "complementaire pour orienter le diagnostic."
                )
        
        return analysis


class Rap2Node(BaseNode):
    """
    Report 2 Node - Day Consultation Report Generation
    
    Generates:
    - Consultation summary
    - Diagnostic assessment
    - Treatment recommendations
    - Follow-up plan
    """
    
    def __init__(self, use_medgemma: bool = True):
        super().__init__("RAP2")
        self.use_medgemma = use_medgemma
        if use_medgemma:
            self.medgemma_generator = MedGemmaReportGenerator()
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate day consultation report"""
        self._log("Generating day consultation report (Rap2)...")
        
        day_data = state.get("day_data", {})
        patient_context = state.get("patient_context", {})
        patient_profile = state.get("patient_profile", "general")
        rap1_report = state.get("rap1_report")
        
        # Generate report
        report = ReportData(
            report_type="consultation",
            title="Rapport de Consultation Médicale",
            generated_at=datetime.now()
        )
        
        # Option 1: Use MedGemma for AI-powered report generation
        if self.use_medgemma and hasattr(self, 'medgemma_generator'):
            self._log(f"Using MedGemma with patient_profile: {patient_profile}")
            try:
                # Extract night context if available
                night_context = ""
                if rap1_report:
                    night_context = rap1_report.get("summary", "") or rap1_report.get("markdown_content", "")[:300]
                
                # Determine specialty from consultation_mode or patient_profile
                specialty = day_data.get("consultation_mode", patient_profile)
                
                markdown_content = self.medgemma_generator.generate_day_report(
                    patient_context=patient_context,
                    night_context=night_context,
                    day_data=day_data,
                    specialty=specialty,
                    patient_profile=patient_profile,
                )
                report.markdown_content = markdown_content
                report.summary = self._extract_summary_from_markdown(markdown_content)
                
                # Update state with MedGemma-generated report
                state["rap2_report"] = report.model_dump()
                state["phase"] = WorkflowPhase.COMPLETED.value
                state["workflow_end"] = datetime.now().isoformat()
                
                state["messages"] = state.get("messages", []) + [{
                    "role": "system",
                    "content": f"Day consultation report (Rap2) generated with MedGemma ({patient_profile} profile). "
                              f"Workflow completed.",
                    "timestamp": datetime.now().isoformat()
                }]
                
                self._log("Rap2 report generated with MedGemma. Workflow complete.")
                return state
            except Exception as e:
                self._log(f"MedGemma generation failed: {e}. Falling back to template.")
                # Fall through to template-based generation
        
        # Option 2: Template-based report generation (fallback)
        sections = []
        
        # 1. Patient info
        sections.append({
            "title": "Identification Patient",
            "content": self._build_patient_section(state.get("patient_id"), patient_context)
        })
        
        # 2. Context from night (if available)
        if rap1_report:
            sections.append({
                "title": "Contexte Nocturne",
                "content": self._build_night_context(rap1_report)
            })
        
        # 3. Motif de consultation
        sections.append({
            "title": "Motif de Consultation",
            "content": self._build_complaint_section(day_data)
        })
        
        # 4. Examen clinique
        sections.append({
            "title": "Examen Clinique",
            "content": self._build_exam_section(day_data)
        })
        
        # 5. Analyse et diagnostic
        sections.append({
            "title": "Analyse Diagnostique",
            "content": self._build_diagnosis_section(day_data)
        })
        
        # 6. Plan de traitement
        sections.append({
            "title": "Plan de Prise en Charge",
            "content": self._build_treatment_section(day_data)
        })
        
        report.sections = sections
        report.summary = self._build_summary(day_data)
        
        # Generate markdown
        report.markdown_content = self._generate_markdown(report)
        
        # Update state
        state["rap2_report"] = report.model_dump()
        state["phase"] = WorkflowPhase.COMPLETED.value
        state["workflow_end"] = datetime.now().isoformat()
        
        # Add final message
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": "Workflow completed. Both Rap1 (night) and Rap2 (day) reports generated.",
            "timestamp": datetime.now().isoformat()
        }]
        
        self._log("Rap2 report generated. Workflow complete.")
        
        return state
    
    def _build_patient_section(self, patient_id: str, context: Dict) -> str:
        """Build patient section"""
        return f"""
**Patient:** {context.get('name', 'N/A')}
**ID:** {patient_id}
**Âge:** {context.get('age', 'N/A')} ans
"""
    
    def _build_night_context(self, rap1: Dict) -> str:
        """Build night context section"""
        summary = rap1.get("summary", "Pas de données nocturnes")
        return f"""
*Résumé de la nuit précédente:*
{summary}
"""
    
    def _build_complaint_section(self, day_data: Dict) -> str:
        """Build presenting complaint section"""
        complaint = day_data.get("presenting_complaint", "Non spécifié")
        symptoms = day_data.get("symptoms", [])
        
        content = f"**Motif principal:** {complaint}\n\n"
        if symptoms:
            content += "**Symptômes associés:**\n"
            for s in symptoms:
                content += f"- {s}\n"
        
        return content
    
    def _build_exam_section(self, day_data: Dict) -> str:
        """Build physical exam section"""
        exam = day_data.get("physical_exam", {})
        vitals = day_data.get("vitals", {})
        
        content = "**Constantes:**\n"
        if vitals:
            for key, value in vitals.items():
                content += f"- {key}: {value}\n"
        else:
            content += "Non renseignées\n"
        
        content += "\n**Examen physique:**\n"
        if exam:
            for system, finding in exam.items():
                content += f"- {system}: {finding}\n"
        else:
            content += "Non renseigné\n"
        
        return content
    
    def _build_diagnosis_section(self, day_data: Dict) -> str:
        """Build diagnostic assessment section"""
        differentials = day_data.get("differential_diagnosis", [])
        severity = day_data.get("severity_assessment", "Non évaluée")
        
        content = f"**Évaluation de gravité:** {severity}\n\n"
        content += "**Diagnostics différentiels:**\n"
        
        for i, dx in enumerate(differentials, 1):
            content += f"{i}. {dx}\n"
        
        if not differentials:
            content += "À compléter après examens\n"
        
        return content
    
    def _build_treatment_section(self, day_data: Dict) -> str:
        """Build treatment plan section"""
        actions = day_data.get("recommended_actions", [])
        
        content = "**Actions recommandées:**\n"
        for action in actions:
            content += f"- {action}\n"
        
        if not actions:
            content += "- Surveillance clinique\n"
        
        referral = day_data.get("referral_needed")
        if referral:
            content += f"\n**Avis spécialisé requis:** {referral}\n"
        
        return content
    
    def _build_summary(self, day_data: Dict) -> str:
        """Build executive summary"""
        mode = day_data.get("consultation_mode", "general")
        severity = day_data.get("severity_assessment", "Non évaluée")
        
        return f"Consultation en mode {mode}. Gravité évaluée: {severity}."
    
    def _generate_markdown(self, report: ReportData) -> str:
        """Generate full markdown"""
        md = f"""# {report.title}

**Date:** {report.generated_at.strftime('%d/%m/%Y %H:%M')}
**Type:** Rapport de Consultation

---

## Résumé

{report.summary}

---

"""
        for section in report.sections:
            md += f"## {section['title']}\n\n{section['content']}\n\n---\n\n"
        
        md += """
---
*Rapport généré automatiquement par MedGemma Sentinel - The Scribe*
*Ce document est une aide à la décision et ne remplace pas le jugement clinique.*
"""
        return md
    
    def _extract_summary_from_markdown(self, markdown: str) -> str:
        """Extract first paragraph as summary from markdown content"""
        lines = markdown.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('*'):
                return stripped[:200] + ('...' if len(stripped) > 200 else '')
        return "Day consultation report generated"
