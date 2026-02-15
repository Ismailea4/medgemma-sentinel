#!/usr/bin/env python3
"""
Smoke test for MedGemma deployment.
Quick validation that core functionality works.
Run this before deployment or after server restart.
"""

import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from medgemma_client import MedGemmaAgent
import requests


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_pass(test_name):
    """Print a passing test."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {test_name}")


def print_fail(test_name, error):
    """Print a failing test."""
    print(f"{Colors.RED}✗{Colors.RESET} {test_name}")
    print(f"  {Colors.RED}Error: {error}{Colors.RESET}")


def print_skip(test_name, reason):
    """Print a skipped test."""
    print(f"{Colors.YELLOW}⊘{Colors.RESET} {test_name} - {reason}")


def test_server_health():
    """Test 1: Check if server is running and responding."""
    test_name = "Server Health Check"
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code in (200, 404):  # 404 is ok, means server is up
            print_pass(test_name)
            return True
        else:
            print_fail(test_name, f"Unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_fail(test_name, "Cannot connect to server. Is it running on port 8080?")
        return False
    except Exception as e:
        print_fail(test_name, str(e))
        return False


def test_model_loaded():
    """Test 2: Check if model is loaded."""
    test_name = "Model Loaded Check"
    try:
        response = requests.get("http://localhost:8080/v1/models", timeout=5)
        data = response.json()
        
        # Check if medgemma model is in the list
        models = data.get("data", [])
        if any(m.get("id") == "medgemma" for m in models):
            print_pass(test_name)
            return True
        else:
            print_fail(test_name, "Model 'medgemma' not found in loaded models")
            return False
    except Exception as e:
        print_fail(test_name, str(e))
        return False


def test_basic_chat():
    """Test 3: Basic chat completion."""
    test_name = "Basic Chat Completion"
    try:
        agent = MedGemmaAgent()
        start_time = time.time()
        
        response = agent.generate_clinical_text(
            "What is hypertension?",
            system_prompt="Answer in one sentence.",
            max_tokens=100,
            temperature=0.1,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        # Check if response is reasonable
        if response and len(response) > 10 and "hypertension" in response.lower():
            print_pass(f"{test_name} ({elapsed:.1f}s)")
            return True
        else:
            print_fail(test_name, f"Invalid response: {response[:100]}")
            return False
    except Exception as e:
        print_fail(test_name, str(e))
        return False


def test_medical_reasoning():
    """Test 4: Medical reasoning capability."""
    test_name = "Medical Reasoning"
    try:
        agent = MedGemmaAgent()
        start_time = time.time()
        
        response = agent.generate_clinical_text(
            "Patient has chest pain and shortness of breath. What is urgent?",
            system_prompt="You are an ER triage nurse. Answer in 1-2 sentences.",
            max_tokens=150,
            temperature=0.1,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        # Check if response mentions cardiac concern
        keywords = ["cardiac", "heart", "myocardial", "emergency", "urgent"]
        has_medical_reasoning = any(kw in response.lower() for kw in keywords)
        
        if has_medical_reasoning and len(response) > 20:
            print_pass(f"{test_name} ({elapsed:.1f}s)")
            return True
        else:
            print_fail(test_name, "Response lacks appropriate medical reasoning")
            return False
    except Exception as e:
        print_fail(test_name, str(e))
        return False


def test_json_mode():
    """Test 5: JSON extraction mode."""
    test_name = "JSON Mode (Structured Output)"
    try:
        agent = MedGemmaAgent()
        start_time = time.time()
        
        result = agent.generate_strict_json(
            "Extract symptoms: Patient has fever and cough.",
            max_tokens=200,
            temperature=0.1,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        # Verify it's valid JSON with expected structure
        if isinstance(result, dict) and "symptoms" in result:
            symptoms = result.get("symptoms", [])
            if isinstance(symptoms, list) and len(symptoms) > 0:
                print_pass(f"{test_name} ({elapsed:.1f}s)")
                print(f"  → Extracted: {json.dumps(result)}")
                return True
        
        print_fail(test_name, f"Invalid JSON structure: {result}")
        return False
    except json.JSONDecodeError:
        print_fail(test_name, "Response is not valid JSON")
        return False
    except Exception as e:
        print_fail(test_name, str(e))
        return False


def test_response_time():
    """Test 6: Response time is acceptable."""
    test_name = "Response Time Check"
    try:
        agent = MedGemmaAgent()
        
        # Make 3 quick requests and average
        times = []
        for _ in range(3):
            start = time.time()
            agent.generate_clinical_text(
                "List one symptom of flu.",
                max_tokens=50,
                temperature=0.1,
                timeout=15
            )
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        
        # Should be under 5 seconds on average for short queries
        if avg_time < 5.0:
            print_pass(f"{test_name} (avg: {avg_time:.2f}s)")
            return True
        else:
            print_fail(test_name, f"Average response time too slow: {avg_time:.2f}s")
            return False
    except Exception as e:
        print_fail(test_name, str(e))
        return False


def test_stop_sequences():
    """Test 7: Stop sequences prevent runaway generation."""
    test_name = "Stop Sequences"
    try:
        agent = MedGemmaAgent()
        
        response = agent.generate_clinical_text(
            "What is the primary symptom of hypothyroidism?",
            system_prompt="Answer in 2-3 sentences.",
            max_tokens=200,
            temperature=0.2,
            timeout=45
        )
        
        # Check that response doesn't have suspicious repetition
        lines = response.split('\n')
        unique_lines = set(lines)
        repetition_ratio = len(unique_lines) / max(len(lines), 1)
        
        # Also check total length is reasonable (not truncated mid-word excessively)
        if repetition_ratio > 0.5 and len(response) > 10:
            print_pass(test_name)
            return True
        else:
            print_fail(test_name, "Detected excessive repetition in output")
            return False
    except Exception as e:
        print_fail(test_name, str(e))
        return False


def run_smoke_tests():
    """Run all smoke tests and report results."""
    print_header("MedGemma Smoke Test Suite")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: http://localhost:8080\n")
    
    tests = [
        ("Server Health", test_server_health),
        ("Model Loaded", test_model_loaded),
        ("Basic Chat", test_basic_chat),
        ("Medical Reasoning", test_medical_reasoning),
        ("JSON Mode", test_json_mode),
        ("Response Time", test_response_time),
        ("Stop Sequences", test_stop_sequences),
    ]
    
    results = []
    total_start = time.time()
    
    for test_desc, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Tests interrupted by user.{Colors.RESET}")
            sys.exit(1)
        except Exception as e:
            print_fail(test_desc, f"Unexpected error: {e}")
            results.append(False)
    
    total_time = time.time() - total_start
    
    # Summary
    passed = sum(results)
    failed = len(results) - passed
    
    print_header("Summary")
    print(f"Total time: {total_time:.1f}s")
    print(f"Tests run: {len(results)}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All tests passed! System is ready.{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ {failed} test(s) failed. Check logs above.{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    exit_code = run_smoke_tests()
    sys.exit(exit_code)
