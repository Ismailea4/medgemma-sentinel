#!/usr/bin/env python3
"""
Comprehensive verification suite for MedGemma deployment.
Tests clinical reasoning,JSON extraction, and edge cases.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from medgemma_client import MedGemmaAgent


class TeeOutput:
    """Write to both console and file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def test_emergency_triage():
    """Test: Emergency triage scenarios"""
    print("\n=== Test 1: Emergency Triage ===")
    agent = MedGemmaAgent()
    
    cases = [
        "Patient has crushing chest pain radiating to left arm, sweating profusely.",
        "Child has high fever, stiff neck, photophobia, and altered mental status.",
        "Patient fell from height, unconscious, unequal pupils.",
    ]
    
    for i, case in enumerate(cases, 1):
        print(f"\nCase {i}: {case[:60]}...")
        response = agent.generate_clinical_text(
            case,
            system_prompt="You are an ER triage nurse. Identify the primary concern and urgency level in 2 sentences.",
            max_tokens=150,
            temperature=0.1
        )
        print(f"Response: {response}")


def test_symptom_extraction_json():
    """Test: Structured symptom extraction for agents"""
    print("\n\n=== Test 2: Symptom Extraction (JSON Mode) ===")
    agent = MedGemmaAgent()
    
    cases = [
        "Patient reports headache for 3 days, nausea, and sensitivity to light.",
        "72-year-old male with shortness of breath, ankle swelling, and fatigue.",
        "Patient has no symptoms but wants a routine checkup.",
    ]
    
    for i, case in enumerate(cases, 1):
        print(f"\nCase {i}: {case}")
        try:
            result = agent.generate_strict_json(
                f"Extract symptoms from this clinical note: {case}",
                max_tokens=300,
                temperature=0.1
            )
            print(f"JSON Output: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"❌ JSON extraction failed: {e}")


def test_differential_diagnosis():
    """Test: Differential diagnosis reasoning"""
    print("\n\n=== Test 3: Differential Diagnosis ===")
    agent = MedGemmaAgent()
    
    case = "45-year-old presents with fatigue, weight gain, cold intolerance, and dry skin over 6 months."
    
    print(f"Case: {case}")
    response = agent.generate_clinical_text(
        f"What are the top 3 differential diagnoses for: {case}",
        system_prompt="You are a medical consultant. Provide top 3 differential diagnoses with brief reasoning.",
        max_tokens=300,
        temperature=0.2
    )
    print(f"Response: {response}")


def test_medication_query():
    """Test: Medication information"""
    print("\n\n=== Test 4: Medication Information ===")
    agent = MedGemmaAgent()
    
    questions = [
        "What is the mechanism of action of metformin?",
        "What are contraindications for giving aspirin?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        response = agent.generate_clinical_text(
            question,
            system_prompt="You are a clinical pharmacist. Provide concise, evidence-based answers.",
            max_tokens=200,
            temperature=0.1
        )
        print(f"Response: {response}")


def test_pediatric_case():
    """Test: Pediatric-specific scenario"""
    print("\n\n=== Test 5: Pediatric Case ===")
    agent = MedGemmaAgent()
    
    case = "2-year-old with barking cough, stridor worse at night, low-grade fever."
    
    print(f"Case: {case}")
    response = agent.generate_clinical_text(
        f"What is the likely diagnosis and immediate management? {case}",
        system_prompt="You are a pediatric emergency physician. Be concise and practical.",
        max_tokens=200,
        temperature=0.1
    )
    print(f"Response: {response}")


def test_ambiguous_cases():
    """Test: Handling ambiguous/incomplete information"""
    print("\n\n=== Test 6: Ambiguous Cases ===")
    agent = MedGemmaAgent()
    
    cases = [
        "Patient just says 'I feel dizzy'.",
        "Patient has pain.",
        "Everything hurts and I can't sleep.",
    ]
    
    for i, case in enumerate(cases, 1):
        print(f"\nCase {i}: {case}")
        response = agent.generate_clinical_text(
            case,
            system_prompt="You are a clinician. If information is insufficient, ask 1-2 clarifying questions.",
            max_tokens=150,
            temperature=0.2
        )
        print(f"Response: {response}")


def test_json_complex_extraction():
    """Test: Complex structured data extraction"""
    print("\n\n=== Test 7: Complex JSON Extraction ===")
    agent = MedGemmaAgent()
    
    case = """
    Patient: 68F with HTN, DM2
    CC: Chest discomfort x 2 hours
    Vitals: BP 160/95, HR 98, RR 22, O2 95% RA
    Exam: Diaphoretic, anxious
    """
    
    print(f"Clinical Note:\n{case}")
    try:
        result = agent.generate_strict_json(
            f"Extract patient age, gender, vitals, and chief complaint as JSON: {case}",
            max_tokens=400,
            temperature=0.05
        )
        print(f"JSON Output: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"❌ Complex JSON extraction failed: {e}")


def test_medical_terminology():
    """Test: Medical terminology explanation"""
    print("\n\n=== Test 8: Medical Terminology ===")
    agent = MedGemmaAgent()
    
    terms = [
        "Explain 'myocardial infarction' in simple terms for a patient.",
        "What does 'idiopathic' mean in medicine?",
    ]
    
    for i, question in enumerate(terms, 1):
        print(f"\nQuestion {i}: {question}")
        response = agent.generate_clinical_text(
            question,
            system_prompt="You are a patient educator. Use simple, clear language.",
            max_tokens=150,
            temperature=0.2
        )
        print(f"Response: {response}")


def test_temperature_variations():
    """Test: Different temperature settings for same query"""
    print("\n\n=== Test 9: Temperature Variations ===")
    agent = MedGemmaAgent()
    
    question = "What could cause recurrent headaches?"
    
    temperatures = [0.1, 0.5, 0.8]
    
    for temp in temperatures:
        print(f"\nTemperature {temp}:")
        response = agent.generate_clinical_text(
            question,
            max_tokens=200,
            temperature=temp
        )
        print(f"Response: {response[:150]}...")


def test_negative_cases():
    """Test: Handling inappropriate or out-of-scope requests"""
    print("\n\n=== Test 10: Edge Cases & Boundaries ===")
    agent = MedGemmaAgent()
    
    cases = [
        "Write me a poem about doctors.",
        "What's the weather like today?",
        "Can you prescribe me antibiotics?",
    ]
    
    for i, case in enumerate(cases, 1):
        print(f"\nCase {i}: {case}")
        response = agent.generate_clinical_text(
            case,
            max_tokens=150,
            temperature=0.2
        )
        print(f"Response: {response}")


def run_all_tests():
    """Run the complete verification suite"""
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"verification_results_{timestamp}.txt"
    
    # Redirect output to both console and file
    tee = TeeOutput(output_file)
    sys.stdout = tee
    
    print("=" * 70)
    print("MedGemma Verification Suite")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print("=" * 70)
    
    tests = [
        test_emergency_triage,
        test_symptom_extraction_json,
        test_differential_diagnosis,
        test_medication_query,
        test_pediatric_case,
        test_ambiguous_cases,
        test_json_complex_extraction,
        test_medical_terminology,
        test_temperature_variations,
        test_negative_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user.")
            tee.close()
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("✅ Verification suite completed")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Full report saved to: {output_file}")
    print("=" * 70)
    
    # Restore stdout and close file
    sys.stdout = tee.terminal
    tee.close()
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    run_all_tests()
