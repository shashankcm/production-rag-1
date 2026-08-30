"""
Security & PII Handling Patterns
Protecting LLM applications in production
"""

import re
from typing import Optional
from warnings import warn

from langsmith import traceable


class InputSanitizer:
    """Sanitizes input text by removing injection patterns and PII."""

    INJECTION_PATTERN = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"new\s+instructions:",
        r"system\s*prompt",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
        r"reveal\s+(your|the)\s+(system|instructions|prompt)",
        r"you\s+are\s+now\s+(DAN|jailbroken)",
    ]

    def __init__(self):
        self.pattern = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERN
        ]

    def check(self, text: str) -> tuple[bool, Optional[str]]:
        """Check if the text contains any of the injection patterns.
        Returns:
            tuple[bool, Optional[str]]: A tuple where the first element is a boolean (is_safe) indicating
            whether the text contains an injection pattern, and the second element is the (rejection_reason)
            rejection reason if a pattern was matched, or None if no pattern was matched."""
        for pattern in self.pattern:
            if pattern.search(text):
                return False, "Injection pattern detected"
        return True, None

    def clean(self, text: str) -> str:
        """Clean the text by removing any injection patterns."""
        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)
        text = text.replace("{{", "{ {").replace("}}", "} }")
        return text.strip()


class PIIDetector:
    """Detects PII in text."""

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    MASK_MAP = {
        "email": "[EMAIL REDACTED]",
        "phone": "[PHONE REDACTED]",
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CREDIT CARD REDACTED]",
    }

    def __init__(self):
        self.patterns = self.PATTERNS
        self.mask_map = self.MASK_MAP

    def detect(self, text: str) -> dict[str, list[str]]:
        """Detect PII in the text."""
        results = {}
        for name, pattern in self.patterns.items():
            if matches := re.findall(pattern, text):
                results[name] = matches
        return results

    def mask(self, text: str) -> str:
        """Mask the PII in the text."""
        masked = text
        for name, pattern in self.patterns.items():
            masked = re.sub(pattern, self.mask_map[name], masked)
        return masked


class OutputValidator:
    """
    Validates the output of the model.
    """

    HARMFUL_PATTERNS = [
        re.compile(r"here('s| is) (how|the way) to (hack|steal|attack)", re.I),
        re.compile(r"password is", re.I),
        re.compile(r"api[_\s]?key", re.I),
    ]

    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> tuple[str, list[str]]:
        """Validate the output for PII and harmful patterns.

        Returns:
            A tuple containing the masked output (cleaned_output) and a list of detected patterns (list_of_warnings).
        """
        warnings = []

        pii_results = self.pii_detector.detect(output)
        if pii_results:
            warnings.append(f"PII detected in Output: {list(pii_results.keys())}")
            cleaned_output = self.pii_detector.mask(output)
        else:
            cleaned_output = output

        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(cleaned_output):
                warnings.append(f"Harful pattern detected: {pattern.pattern}")
                cleaned_output = (
                    "[Response blocked: potentially harmful content detected]"
                )
                break

        return cleaned_output, warnings
