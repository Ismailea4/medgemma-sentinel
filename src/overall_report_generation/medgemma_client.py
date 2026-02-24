import json
import re
from typing import Any, Dict, Optional

import requests


JSON_GBNF = r'''
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws (
    string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::= "[" ws (
    value
    ("," ws value)*
  )? "]" ws
string ::= "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws
number ::= ("-"? ([0-9]+) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?) ws
ws ::= ([ \t\n]*)
'''


class MedGemmaAgent:
    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self.base_url = base_url.rstrip("/")

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        if "content" in payload:
            return payload.get("content", "")
        choices = payload.get("choices", [])
        if choices:
            return choices[0].get("text") or choices[0].get("message", {}).get("content", "")
        return ""

    def _strip_thoughts(self, text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"thought\s*:\s*.*?(\n\n|$)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    def generate_clinical_text(
        self,
        prompt: str,
        system_prompt: str = "You are a medical assistant. Provide clear, evidence-based answers. Be concise.",
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: int = 30,
    ) -> str:
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": "medgemma",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["<end_of_turn>", "\n\n\n", "User:", "Question:"],
            "frequency_penalty": 0.3,
        }

        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        raw = self._extract_text(response.json())
        return self._strip_thoughts(raw)

    def generate_strict_json(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/completion"
        full_prompt = (
            "<start_of_turn>user\n"
            "You are an API that outputs JSON only.\n\n"
            f"{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        payload = {
            "prompt": full_prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "grammar": JSON_GBNF,
            "stop": ["<end_of_turn>", "</s>"],
        }

        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        content = self._extract_text(response.json())
        return json.loads(content)


if __name__ == "__main__":
    agent = MedGemmaAgent()
    print("--- Chat Test ---")
    print(agent.generate_clinical_text("List 3 symptoms of flu."))

    print("\n--- JSON Test ---")
    data = agent.generate_strict_json(
        "Extract symptoms: Patient has high fever (39C) and dry cough."
    )
    print(data)
