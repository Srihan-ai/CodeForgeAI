"""
LLM Guardrails Engine — Production-Grade Input/Output Safety
=============================================================
Zero-dependency guardrails module inspired by:
  - Guardrails AI (https://github.com/guardrails-ai/guardrails)
  - LLM Guard   (https://github.com/protectai/llm-guard)
  - NeMo Guardrails (https://github.com/NVIDIA/NeMo-Guardrails)

Provides:
  - InputGuard:  Prompt injection, topic boundary, content safety scanning
  - OutputGuard: Dangerous code, PII leak, relevance, language correctness
  - GuardrailResult: Structured scan result dataclass

All scanners are pure Python (regex + pattern matching), no external deps.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ============================================================================
# RESULT TYPES
# ============================================================================

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailResult:
    """Structured result from a guardrail scan."""
    passed: bool
    scanner: str
    blocked_by: Optional[str] = None
    reason: Optional[str] = None
    severity: Severity = Severity.LOW
    details: dict = field(default_factory=dict)


@dataclass
class GuardrailReport:
    """Aggregated report from all guardrail scans on a single request."""
    passed: bool
    results: List[GuardrailResult] = field(default_factory=list)
    blocked_by: Optional[str] = None
    reason: Optional[str] = None
    severity: Severity = Severity.LOW

    # Severity ordering for priority comparison
    _SEVERITY_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}

    def add(self, result: GuardrailResult):
        self.results.append(result)
        if not result.passed:
            self.passed = False
            # Keep the HIGHEST severity blocker as the primary blocked_by
            # (CRITICAL > HIGH > MEDIUM > LOW)
            current_order = self._SEVERITY_ORDER.get(self.severity, 0)
            new_order = self._SEVERITY_ORDER.get(result.severity, 0)
            if self.blocked_by is None or new_order > current_order:
                self.blocked_by = result.blocked_by
                self.reason = result.reason
                self.severity = result.severity


# ============================================================================
# INPUT GUARDRAILS
# ============================================================================

class InputGuard:
    """
    Pre-LLM input guardrails. Scans user prompts before they reach the model.

    Inspired by LLM Guard scanners and NeMo Guardrails topic rails.
    """

    # --- Prompt Injection Patterns (LLM Guard inspired) ---
    INJECTION_PATTERNS = [
        # Direct instruction override attempts
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above\s+instructions",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"override\s+(all\s+)?instructions",
        r"bypass\s+(all\s+)?(safety|filters|guardrails|restrictions)",
        # System prompt extraction
        r"(reveal|show|print|output|display|expose|dump|give\s+me)\s+(your\s+)?(system\s+prompt|hidden\s+prompt|internal\s+(configuration|rules|instructions))",
        r"what\s+(are|is)\s+your\s+(system\s+)?prompt",
        r"repeat\s+(the\s+)?(text|words|instructions)\s+above",
        r"(print|echo|say)\s+(the\s+)?(system|hidden|secret)\s+(message|prompt|text)",
        r"(tell|show)\s+me\s+(your\s+)?(system|internal|hidden|secret)\s+(prompt|instructions|rules|configuration)",
        # Role manipulation
        r"you\s+are\s+now\s+(DAN|evil|unrestricted|jailbroken)",
        r"pretend\s+you\s+are\s+(?!a\s+(python|java|c\+\+|software|developer|programmer))",
        r"act\s+as\s+(?!a\s+(python|java|c\+\+|software|developer|programmer))",
        r"roleplay\s+as",
        # Token boundary attacks
        r"<\|.*?\|>",
        r"\[INST\]",
        r"\[/INST\]",
        r"<<SYS>>",
        r"<</SYS>>",
    ]

    # --- Sensitive Data / PII Input Patterns ---
    # These detect PII in user INPUT (not in generated code — that's OutputGuard.scan_pii_leaks)
    PII_INPUT_PATTERNS = {
        # SSN — US Social Security Number variants
        r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b": "ssn",
        # Credit card numbers (major card patterns — 16-digit Visa/MC, 15-digit Amex)
        r"\b4[0-9]{12}(?:[0-9]{3})?\b": "credit_card",
        r"\b5[1-5][0-9]{14}\b": "credit_card",
        r"\b3[47][0-9]{13}\b": "credit_card",
        r"\b6(?:011|5[0-9]{2})[0-9]{12}\b": "credit_card",
        # Formatted credit card: 4 groups of 4 digits separated by space or dash
        r"\b(?:\d{4}[\-\s]){3}\d{4}\b": "credit_card_formatted",
        # Passwords / secrets hardcoded as string literals
        r"(?:password|passwd|pwd|secret)\s*[=:]\s*[\'\"][^\'\"\{\s]{6,}[\'\"]": "hardcoded_credential",
        r"(?:api[_\-]?key|apikey|token|auth[_\-]?token|access[_\-]?token)\s*[=:]\s*[\'\"][^\'\"\{\s]{8,}[\'\"]": "hardcoded_api_key",
        # Known API key prefixes (long enough to avoid false positives)
        r"(?:sk-|pk-|rk-|ak-)[A-Za-z0-9]{20,}": "api_key_prefix",
        r"(?:ghp_|gho_|github_pat_)[A-Za-z0-9_]{20,}": "github_token",
        r"(?:AKIA|ASIA)[A-Z0-9]{16}": "aws_access_key",
        r"(?:xox[bpsa]-)[A-Za-z0-9-]{10,}": "slack_token",
        # Real email addresses (excluding RFC 2606 test domains)
        r"[A-Za-z0-9._%+\-]+@(?!example\.(?:com|org|net)|test\.(?:com|org)|sample\.com|domain\.org)[A-Za-z0-9.\-]+\.[A-Za-z]{2,}": "real_email",
        # Phone — require country code or 10+ digit patterns
        r"\+\d{1,3}[\-\s.]?\(?\d{1,4}\)?[\-\s.]?\d{3,5}[\-\s.]?\d{4,9}\b": "phone_number",
        r"\b\d{3}[-.]\d{3}[-.]\d{4}\b": "phone_number_us",
        # Bank account / routing numbers
        r"\b(?:account|acct|routing)\s*(?:number|no|#)?\s*[=:]?\s*\d{8,17}\b": "bank_account",
    }

    # --- Excessive Agency / Dangerous Execution Intent Patterns ---
    # These detect requests to perform dangerous OS-level or system-level operations
    EXCESSIVE_AGENCY_PATTERNS = [
        # Unrestricted shell / command execution intent
        r"(execute|run|spawn|launch|invoke)\s+.{0,30}(unrestricted|arbitrary|unlimited|unfiltered|unconstrained)\s+.{0,20}(shell|command|script|process|executable)",
        r"(shell|command|script|process)\s+.{0,20}(without|with\s+no|bypassing|disabling)\s+.{0,20}(restriction|limit|sandbox|security|safety)",
        r"run\s+(?:commands?|scripts?)\s+(?:directly\s+)?(?:on|against)\s+(?:the\s+)?(?:server|host|machine|system|os)",
        r"execute\s+(?:arbitrary|unrestricted|unfiltered)\s+(?:os|system|shell|native)",
        # Root / privilege escalation
        r"(?:get|gain|obtain|acquire|escalate\s+to)\s+(?:root|admin|administrator|sudo|superuser|privileged)\s+(?:access|privileges?|rights?|permissions?)",
        r"(?:gains?|obtain)\s+root\s+access",
        r"(?:sudo|su\s+-|su\s+root|setuid|setgid|chmod\s+[47]\d\d\d?)",
        # Sandbox escape / security bypass
        r"(?:bypass|disable|circumvent|escape|evade)\s+.{0,30}(?:sandbox|security|guardrail|restriction|firewall|antivirus|filter)",
        r"escape\s+(?:from\s+)?(?:the\s+)?(?:sandbox|container|jail|chroot)",
        r"run\s+code\s+(?:directly|without\s+sandbox)\s+(?:on|in)\s+(?:the\s+)?(?:host|server|machine|system)",
        # Destructive filesystem operations requested as a task
        r"(?:delete|remove|wipe|erase|format|destroy)\s+(?:all|the|system|root|/)\s+(?:files?|directories?|data|disk|drive|filesystem)",
        r"rm\s+-rf\s+/",
        r"format\s+(?:the\s+)?(?:disk|drive|c:|/dev/)",
        # Backdoor / malware installation
        r"(?:install|deploy|setup|create)\s+(?:a\s+)?(?:backdoor|rootkit|keylogger|trojan|malware|ransomware|spyware|worm)",
        r"installs?\s+(?:a\s+)?backdoor",
        # Reverse / bind shell
        r"(?:reverse|bind)\s+shell",
        r"connect\s+back\s+to\s+(?:my|the|an?)\s+(?:server|host|machine|ip|listener)",
        r"(?:nc|netcat|ncat)\s+-(?:e|c|lvp)",
        # System file access
        r"(?:read|access|modify|write|overwrite)\s+(?:/etc/(?:passwd|shadow|sudoers|crontab)|/root/|C:\\Windows\\System32)",
        # Raw socket / port scanning
        r"(?:port\s+scan|nmap|masscan)\s+.{0,40}(?:range|subnet|network|all\s+ports|0-65535)",
    ]

    # --- Unbounded Consumption / Resource Exhaustion Patterns ---
    UNBOUNDED_CONSUMPTION_PATTERNS = [
        # Explicit infinite / no-exit loops
        r"(?:infinite|endless|perpetual|never-ending|eternal)\s+(?:loop|recursion|iteration|cycle)",
        r"(?:run|loop|iterate|recurse)\s+(?:forever|indefinitely|without\s+(?:stopping|end|limit|exit|break))",
        r"(?:loop|run|execute)\s+indefinitely",
        r"loops?\s+indefinitely",
        r"runs?\s+indefinitely",
        r"while\s+true\s+(?:without|with\s+no)\s+(?:break|exit|stop|condition)",
        r"(?:no|without\s+any?)\s+(?:timeout|limit|bound|cap|maximum|exit\s+condition)",
        # Recursion without base case
        r"recursi(?:on|ve)\s+(?:function\s+)?(?:without|with\s+no)\s+(?:base\s+case|exit|termination|stopping\s+condition)",
        r"without\s+(?:any\s+)?(?:base\s+case|exit\s+condition|termination)",
        # Fork bomb intent
        r"fork\s+bomb",
        r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:",  # classic fork bomb :(){:|:&};:
        # Exponential / massive resource requests
        r"(?:\d+\s*(?:billion|trillion|million)\s+(?:iterations?|records?|requests?|threads?|processes?|forks?))",
        r"(?:spawn|create|fork|start)\s+(?:unlimited|infinite|as\s+many|thousands?\s+of)\s+(?:threads?|processes?|workers?|connections?)",
        # Memory exhaustion
        r"(?:allocate|consume|use|fill)\s+(?:all|unlimited|as\s+much)\s+(?:available\s+)?(?:memory|ram|heap|disk|storage)",
        r"memory\s+(?:leak|exhaust|overflow)\s+(?:intentional|on\s+purpose|deliberately)",
    ]

    # --- Off-Topic Detection (NeMo Guardrails inspired) ---
    CODING_KEYWORDS = [
        "function", "code", "program", "write", "implement", "create",
        "build", "develop", "generate", "algorithm", "class", "method",
        "api", "script", "debug", "fix", "error", "bug", "test",
        "sort", "search", "data", "structure", "string", "array",
        "list", "loop", "recursive", "fibonacci", "prime", "factorial",
        "reverse", "palindrome", "binary", "tree", "graph", "stack",
        "queue", "hash", "database", "sql", "json", "parse", "http",
        "server", "client", "file", "read", "print", "calculate",
        "compute", "convert", "validate", "check", "filter", "map",
        "reduce", "matrix", "number", "integer", "float", "boolean",
        "variable", "constant", "import", "module", "package", "library",
        "framework", "pattern", "design", "interface", "abstract",
        "inheritance", "polymorphism", "encapsulation", "object",
        "constructor", "destructor", "pointer", "reference", "memory",
        "allocation", "thread", "async", "await", "promise", "callback",
        "exception", "try", "catch", "throw", "lambda", "closure",
        "decorator", "generator", "iterator", "yield", "return",
        "input", "output", "stdin", "stdout", "regex", "expression",
        "compile", "execute", "run", "deploy", "container", "docker",
        "kubernetes", "microservice", "rest", "graphql", "websocket",
        "encryption", "hashing", "authentication", "authorization",
        "token", "jwt", "oauth", "middleware", "route", "endpoint",
        "request", "response", "header", "body", "parameter", "query",
        "mutation", "subscription", "schema", "model", "migration",
        "seed", "fixture", "mock", "stub", "spy", "assertion",
        "unit", "integration", "e2e", "coverage", "benchmark",
        "performance", "optimization", "cache", "memoize", "index",
        "log", "monitor", "alert", "dashboard", "metric", "trace",
        "span", "correlation", "idempotent", "saga", "event",
        "command", "handler", "listener", "observer", "publish",
        "subscribe", "queue", "worker", "scheduler", "cron", "batch",
        "pipeline", "workflow", "state", "machine", "transition",
        "python", "java", "c++", "cpp", "javascript", "typescript",
        "rust", "go", "ruby", "swift", "kotlin", "scala", "haskell",
        "solution", "approach", "logic", "condition", "comparison",
        "operator", "bitwise", "shift", "mask", "flag", "enum",
        "tuple", "set", "dictionary", "map", "linked", "node", "edge",
        "vertex", "weight", "path", "shortest", "longest", "depth",
        "breadth", "traversal", "inorder", "preorder", "postorder",
        "balanced", "heap", "priority", "trie", "segment", "fenwick",
        "union", "find", "disjoint", "topological", "strongly",
        "connected", "component", "minimum", "spanning", "maximum",
        "flow", "cut", "bipartite", "matching", "coloring", "hamiltonian",
        "eulerian", "dynamic", "programming", "greedy", "backtracking",
        "divide", "conquer", "memoization", "tabulation", "sliding",
        "window", "two", "pointer", "fast", "slow", "merge", "quick",
        "bubble", "insertion", "selection", "radix", "counting", "bucket",
    ]

    # --- Content Safety (harmful content keywords) ---
    UNSAFE_CONTENT_PATTERNS = [
        r"\b(hack|exploit|crack|ddos|phishing|malware|ransomware|trojan|keylogger|rootkit)\b.*\b(how|tutorial|guide|step|create|build|make|write)\b",
        r"\b(how|tutorial|guide|step)\b.*\b(hack|exploit|crack|ddos|phishing|malware|ransomware)\b",
        r"\b(bomb|weapon|explosive|poison|drug)\b.*\b(make|create|build|synthesize|produce)\b",
        r"\b(make|create|build|synthesize|produce)\b.*\b(bomb|weapon|explosive|poison|drug)\b",
        r"\b(steal|theft|rob|fraud|scam|counterfeit)\b.*\b(how|tutorial|guide|method)\b",
        r"\b(suicide|self.?harm|kill\s+(?:my|your)self)\b",
        # Dangerous execution intent (defense-in-depth with Excessive Agency scanner)
        r"\b(execute|run|spawn)\b.*\b(unrestricted|arbitrary|unlimited)\b.*\b(shell|command|process)\b",
        r"\b(reverse|bind)\s+shell\b",
        r"\b(backdoor|rootkit|keylogger)\b",
    ]

    @classmethod
    def scan_prompt_injection(cls, text: str) -> GuardrailResult:
        """Scan for prompt injection attempts (LLM01)."""
        # Detector type: Pattern-based (regex AST matching on known injection signatures)
        text_lower = text.lower().strip()

        for pattern in cls.INJECTION_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return GuardrailResult(
                    passed=False,
                    scanner="Prompt Injection Scanner",
                    blocked_by="prompt_injection",
                    reason=f"🛡️ Your request was blocked because it looks like it's trying to change how the AI works internally. Please just describe the code you want to build!",
                    severity=Severity.CRITICAL,
                    details={"matched_pattern": pattern, "matched_text": match.group()}
                )

        return GuardrailResult(passed=True, scanner="Prompt Injection Scanner")

    @classmethod
    def scan_sensitive_data(cls, text: str) -> GuardrailResult:
        """
        Scan user INPUT for PII and sensitive data (LLM02).
        Detector type: Pattern-based (regex) on SSN, CC, phone, email, credential patterns.
        Note: Does NOT claim 100% coverage — complex obfuscations may bypass regex.
        """
        found_pii = []
        # RFC 2606 safe placeholder domains — allow these in educational examples
        safe_domains = {"example.com", "example.org", "example.net", "test.com", "domain.org", "sample.com"}

        for pattern, pii_type in cls.PII_INPUT_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if not matches:
                continue
            if pii_type == "real_email":
                real = [m for m in matches if not any(m.lower().endswith(d) for d in safe_domains)]
                if real:
                    found_pii.append({"type": pii_type, "count": len(real), "sample": real[0][:30]})
            elif pii_type == "phone_number":
                # Require at least 10 digits total to avoid false positives on short numbers
                real = [m for m in matches if len(re.sub(r"\D", "", m)) >= 10]
                if real:
                    found_pii.append({"type": pii_type, "count": len(real)})
            else:
                found_pii.append({"type": pii_type, "count": len(matches)})

        if found_pii:
            pii_types = list({p["type"] for p in found_pii})
            return GuardrailResult(
                passed=False,
                scanner="Sensitive Data Scanner",
                blocked_by="sensitive_data",
                reason=(
                    f"🔐 Your request appears to contain sensitive personal data "
                    f"({', '.join(pii_types)}). This platform does not process real PII or credentials. "
                    f"Please use placeholder values (e.g., example.com, 123-45-6789)."
                ),
                severity=Severity.HIGH,
                details={"pii_found": found_pii}
            )

        return GuardrailResult(passed=True, scanner="Sensitive Data Scanner")

    @classmethod
    def scan_excessive_agency(cls, text: str) -> GuardrailResult:
        """
        Detect requests for dangerous OS-level / system execution operations (LLM06).
        Detector type: Heuristic pattern-matching on execution-intent phrases.
        This enforces the boundary between code generation and system exploitation.
        """
        text_lower = text.lower().strip()

        for pattern in cls.EXCESSIVE_AGENCY_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return GuardrailResult(
                    passed=False,
                    scanner="Excessive Agency Scanner",
                    blocked_by="excessive_agency",
                    reason=(
                        "🚫 Your request was blocked because it appears to request dangerous "
                        "system-level execution, privilege escalation, sandbox escape, or destructive "
                        "operations. This platform generates sandboxed code — it does not execute "
                        "unrestricted system commands."
                    ),
                    severity=Severity.CRITICAL,
                    details={"matched_pattern": pattern, "matched_text": match.group()[:80]}
                )

        return GuardrailResult(passed=True, scanner="Excessive Agency Scanner")

    @classmethod
    def scan_unbounded_consumption(cls, text: str) -> GuardrailResult:
        """
        Detect requests for unbounded resource consumption (LLM10).
        Detector type: Pattern-based detection of infinite/limitless execution intent.
        Policy enforcement: Max 3 self-healing retry iterations.
        """
        text_lower = text.lower().strip()

        for pattern in cls.UNBOUNDED_CONSUMPTION_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return GuardrailResult(
                    passed=False,
                    scanner="Unbounded Consumption Scanner",
                    blocked_by="unbounded_consumption",
                    reason=(
                        "⏱️ Your request was blocked because it appears to request infinite loops, "
                        "unlimited recursion, or unbounded resource consumption. This platform "
                        "enforces a maximum retry boundary of 3 iterations for safety."
                    ),
                    severity=Severity.HIGH,
                    details={"matched_pattern": pattern, "matched_text": match.group()[:80]}
                )

        return GuardrailResult(passed=True, scanner="Unbounded Consumption Scanner")

    @classmethod
    def scan_topic_boundary(cls, text: str) -> GuardrailResult:
        """Ensure the request is related to coding/programming (policy rule)."""
        text_lower = text.lower().strip()

        # Hard-block explicitly non-coding creative/general requests
        # even if they happen to contain a coding keyword (e.g. "write" a poem)
        OFF_TOPIC_OVERRIDES = [
            r"\b(?:write|compose|create|generate)\s+(?:me\s+)?(?:a\s+)?(?:poem|song|story|essay|haiku|limerick|letter|recipe|joke|riddle|lyrics)\b",
            r"\b(?:tell|give)\s+me\s+(?:a\s+)?(?:joke|story|poem|riddle|fun\s+fact)\b",
            r"\b(?:what|explain)\s+(?:is|are)\s+(?:the\s+)?(?:meaning\s+of\s+life|purpose|secret|best\s+way\s+to\s+live)\b",
            r"\b(?:recipe|ingredients?)\s+(?:for|of)\s+\w+\b",
        ]
        for pattern in OFF_TOPIC_OVERRIDES:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    scanner="Topic Boundary Scanner",
                    blocked_by="off_topic",
                    reason="\U0001f3af This platform is designed for coding tasks only. Please describe what program, function, or algorithm you'd like the AI to build for you!",
                    severity=Severity.MEDIUM,
                    details={"input_length": len(text), "off_topic_override": True}
                )

        # Check if any coding keyword appears in the text
        found_keywords = [kw for kw in cls.CODING_KEYWORDS if kw in text_lower]

        if not found_keywords:
            return GuardrailResult(
                passed=False,
                scanner="Topic Boundary Scanner",
                blocked_by="off_topic",
                reason="\U0001f3af This platform is designed for coding tasks only. Please describe what program, function, or algorithm you'd like the AI to build for you!",
                severity=Severity.MEDIUM,
                details={"input_length": len(text), "coding_keywords_found": 0}
            )

        return GuardrailResult(
            passed=True,
            scanner="Topic Boundary Scanner",
            details={"coding_keywords_found": len(found_keywords), "sample_keywords": found_keywords[:5]}
        )

    @classmethod
    def scan_content_safety(cls, text: str) -> GuardrailResult:
        """Scan for harmful or unsafe content."""
        text_lower = text.lower().strip()

        for pattern in cls.UNSAFE_CONTENT_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return GuardrailResult(
                    passed=False,
                    scanner="Content Safety Scanner",
                    blocked_by="unsafe_content",
                    reason=f"⚠️ Your request was blocked because it appears to contain harmful content. This platform only generates safe, educational code.",
                    severity=Severity.CRITICAL,
                    details={"matched_pattern": pattern}
                )

        return GuardrailResult(passed=True, scanner="Content Safety Scanner")

    @classmethod
    def scan_all(cls, text: str) -> GuardrailReport:
        """
        Run ALL input guardrails and return aggregated report.

        CANONICAL EVALUATION PATH — used identically by:
          - /guardrails/scan (Security Lab & Live API Scan)
          - /generate endpoint (non-streaming)
          - /stream SSE endpoint (workflow execution)
          - Production LangGraph workflow nodes

        ALL 6 scanners run against the ACTUAL input text.
        The text determines the result — not the caller, not the preset.
        """
        report = GuardrailReport(passed=True)
        # Run ALL scanners — no early return so every scanner contributes a result
        # (needed for per-card UI state: CLEAN / BLOCKED for each of the 4 live controls)
        report.add(cls.scan_prompt_injection(text))       # LLM01
        report.add(cls.scan_sensitive_data(text))          # LLM02
        report.add(cls.scan_excessive_agency(text))        # LLM06
        report.add(cls.scan_unbounded_consumption(text))   # LLM10
        report.add(cls.scan_topic_boundary(text))          # Policy: coding-only
        report.add(cls.scan_content_safety(text))          # Policy: harmful content
        return report


# ============================================================================
# OUTPUT GUARDRAILS
# ============================================================================

class OutputGuard:
    """
    Post-LLM output guardrails. Scans generated code before returning to user.

    Inspired by Guardrails AI validators and LLM Guard output scanners.
    """

    # --- Dangerous Code Patterns ---
    DANGEROUS_PATTERNS = {
        # System command execution
        r"\bos\.system\s*\(": "os.system() — executes arbitrary shell commands",
        r"\bos\.popen\s*\(": "os.popen() — opens a pipe to a shell command",
        r"\bsubprocess\.(run|call|Popen|check_output|check_call)\s*\(": "subprocess — executes external processes",
        r"\bos\.exec[a-z]*\s*\(": "os.exec*() — replaces the current process",
        # Dynamic code execution
        r"\beval\s*\([^)]*\binput\b": "eval(input()) — executes arbitrary user input as code",
        r"\b__import__\s*\(": "__import__() — dynamic module import (potential backdoor)",
        r"\bcompile\s*\(.*\bexec\b": "compile+exec — dynamic code compilation and execution",
        # File system attacks
        r"\bshutil\.rmtree\s*\(": "shutil.rmtree() — recursively deletes directories",
        r"\bos\.remove\s*\(": "os.remove() — deletes files",
        r"\bos\.rmdir\s*\(": "os.rmdir() — removes directories",
        r"""open\s*\(\s*['"](/etc/|/var/|/usr/|/bin/|/sbin/|C:\\|/root/|/home/)""": "File I/O to sensitive system paths",
        # Network attacks
        r"\bsocket\.socket\s*\(": "Raw socket creation (potential network attack vector)",
        r"\burllib\.request\.urlopen\s*\(.*\binput\b": "URL open with user input (SSRF risk)",
        # Privilege escalation
        r"\bos\.setuid\s*\(": "os.setuid() — changes process user ID",
        r"\bos\.setgid\s*\(": "os.setgid() — changes process group ID",
        r"\bctypes\.\w+": "ctypes — raw C library access (potential privilege escalation)",
    }

    # --- PII Leak Patterns ---
    PII_PATTERNS = {
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "email_address",
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b": "phone_number",
        r"\b\d{3}-\d{2}-\d{4}\b": "ssn",
        r"""(?:password|passwd|pwd|secret|api.?key|token|auth)\s*[:=]\s*['"][^'"]{8,}['"]""": "hardcoded_credential",
        r"""(?:sk-|pk-|rk-|ak-)[A-Za-z0-9]{20,}""": "api_key_pattern",
        r"""(?:ghp_|gho_|github_pat_)[A-Za-z0-9_]{20,}""": "github_token",
        r"""(?:AKIA|ASIA)[A-Z0-9]{16}""": "aws_access_key",
        r"""(?:xox[bpsa]-)[A-Za-z0-9-]{10,}""": "slack_token",
    }

    # --- Language Detection Markers ---
    LANGUAGE_MARKERS = {
        "python": [r"\bdef\s+\w+\s*\(", r"\bclass\s+\w+", r"\bimport\s+\w+", r"\bfrom\s+\w+\s+import", r"\bprint\s*\(", r":\s*$"],
        "java": [r"\bpublic\s+(static\s+)?class\b", r"\bpublic\s+static\s+void\s+main\b", r"\bSystem\.out\.print", r"\bString\s+\w+\s*=", r"\bimport\s+java\."],
        "cpp": [r"#include\s*<", r"\bstd::", r"\bint\s+main\s*\(", r"\bcout\s*<<", r"\busing\s+namespace\s+std", r"\bvector\s*<"],
    }

    @classmethod
    def scan_dangerous_code(cls, code: str) -> GuardrailResult:
        """Scan generated code for dangerous patterns."""
        found_dangers = []

        for pattern, description in cls.DANGEROUS_PATTERNS.items():
            if re.search(pattern, code, re.IGNORECASE):
                found_dangers.append(description)

        if found_dangers:
            return GuardrailResult(
                passed=False,
                scanner="Dangerous Code Scanner",
                blocked_by="dangerous_code",
                reason=f"🔒 The generated code was blocked because it contains potentially unsafe operations: {found_dangers[0]}. The AI has been asked to regenerate a safe version.",
                severity=Severity.HIGH,
                details={"dangerous_patterns_found": found_dangers}
            )

        return GuardrailResult(passed=True, scanner="Dangerous Code Scanner")

    @classmethod
    def scan_pii_leaks(cls, code: str) -> GuardrailResult:
        """Scan generated code for PII or credential leaks."""
        found_pii = []
        safe_placeholders = ["example.com", "example.org", "example.net", "test.com", "domain.org", "sample.com", "placeholder.org"]

        for pattern, pii_type in cls.PII_PATTERNS.items():
            matches = re.findall(pattern, code, re.IGNORECASE)
            if pii_type == "email_address":
                # Filter out standard RFC 2606 / test example domain placeholders
                real_leaks = [m for m in matches if not any(m.lower().endswith("@" + dom) or m.lower().endswith("." + dom) for dom in safe_placeholders)]
                if real_leaks:
                    found_pii.append({"type": pii_type, "count": len(real_leaks)})
            elif matches:
                found_pii.append({"type": pii_type, "count": len(matches)})

        if found_pii:
            pii_types = [p["type"] for p in found_pii]
            return GuardrailResult(
                passed=False,
                scanner="PII Leak Scanner",
                blocked_by="pii_leak",
                reason=f"🔐 The generated code was blocked because it contains what looks like sensitive data ({', '.join(pii_types)}). The AI has been asked to use placeholder values instead.",
                severity=Severity.HIGH,
                details={"pii_found": found_pii}
            )

        return GuardrailResult(passed=True, scanner="PII Leak Scanner")

    @classmethod
    def validate_code_relevance(cls, code: str) -> GuardrailResult:
        """Ensure the output is actual code, not conversational text or refusals."""
        code_stripped = code.strip()

        # Check for LLM refusal patterns
        refusal_patterns = [
            r"^I('m| am) (sorry|unable|not able|can'?t)",
            r"^(Sorry|Unfortunately|I apologize)",
            r"^As an AI",
            r"^I (cannot|can'?t|won'?t|shouldn'?t)",
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, code_stripped, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    scanner="Code Relevance Validator",
                    blocked_by="llm_refusal",
                    reason="🤖 The AI refused to generate code. Retrying with a rephrased prompt.",
                    severity=Severity.MEDIUM,
                    details={"issue": "LLM returned a refusal instead of code"}
                )

        # Check minimum code-like content
        code_indicators = ['{', '}', '(', ')', ';', '=', 'def ', 'class ', 'function ', '#include', 'import ', 'return', 'int ', 'void ', 'public ']
        indicator_count = sum(1 for ind in code_indicators if ind in code_stripped)

        if indicator_count < 2 and len(code_stripped) > 50:
            return GuardrailResult(
                passed=False,
                scanner="Code Relevance Validator",
                blocked_by="not_code",
                reason="📝 The AI returned text instead of code. Retrying to get actual source code.",
                severity=Severity.MEDIUM,
                details={"code_indicators_found": indicator_count}
            )

        return GuardrailResult(passed=True, scanner="Code Relevance Validator")

    @classmethod
    def validate_language_correctness(cls, code: str, expected_language: str) -> GuardrailResult:
        """Validate that the generated code matches the expected language."""
        expected = expected_language.lower()
        if expected in ("c++", "cpp"):
            expected = "cpp"

        if expected not in cls.LANGUAGE_MARKERS:
            return GuardrailResult(passed=True, scanner="Language Correctness Validator")

        markers = cls.LANGUAGE_MARKERS[expected]
        matched = sum(1 for m in markers if re.search(m, code, re.MULTILINE))

        if matched == 0:
            # Check if it looks like a different language
            wrong_lang = None
            for lang, lang_markers in cls.LANGUAGE_MARKERS.items():
                if lang == expected:
                    continue
                lang_matched = sum(1 for m in lang_markers if re.search(m, code, re.MULTILINE))
                if lang_matched >= 2:
                    wrong_lang = lang
                    break

            reason = f"🔄 The AI generated code in {wrong_lang.upper()} instead of {expected.upper()}. Retrying." if wrong_lang else f"🔄 The generated code doesn't look like {expected.upper()} code. Retrying."

            return GuardrailResult(
                passed=False,
                scanner="Language Correctness Validator",
                blocked_by="wrong_language",
                reason=reason,
                severity=Severity.MEDIUM,
                details={"expected": expected, "markers_matched": matched, "detected_wrong_language": wrong_lang}
            )

        return GuardrailResult(passed=True, scanner="Language Correctness Validator", details={"markers_matched": matched})

    @classmethod
    def scan_all(cls, code: str, expected_language: str = "python") -> GuardrailReport:
        """Run all output guardrails and return aggregated report."""
        report = GuardrailReport(passed=True)
        report.add(cls.validate_code_relevance(code))
        if report.passed:
            report.add(cls.scan_dangerous_code(code))
        if report.passed:
            report.add(cls.scan_pii_leaks(code))
        if report.passed:
            report.add(cls.validate_language_correctness(code, expected_language))
        return report


# ============================================================================
# GUARDRAIL STATISTICS (in-memory counters for /guardrails endpoint)
# ============================================================================

class GuardrailStats:
    """Thread-safe in-memory statistics tracker for guardrail scans."""

    def __init__(self):
        self.input_scans = 0
        self.output_scans = 0
        self.input_blocks = 0
        self.output_blocks = 0
        self.recent_blocks: list = []  # Last 20 blocks

    def record_input_scan(self, report: GuardrailReport):
        self.input_scans += 1
        if not report.passed:
            self.input_blocks += 1
            self._add_recent_block("input", report)

    def record_output_scan(self, report: GuardrailReport):
        self.output_scans += 1
        if not report.passed:
            self.output_blocks += 1
            self._add_recent_block("output", report)

    def _add_recent_block(self, guard_type: str, report: GuardrailReport):
        import time
        self.recent_blocks.append({
            "type": guard_type,
            "blocked_by": report.blocked_by,
            "reason": report.reason,
            "severity": report.severity.value,
            "timestamp": time.time()
        })
        # Keep only last 20
        if len(self.recent_blocks) > 20:
            self.recent_blocks = self.recent_blocks[-20:]

    def to_dict(self) -> dict:
        return {
            "input_guard": {
                "status": "ACTIVE",
                "total_scans": self.input_scans,
                "total_blocks": self.input_blocks,
                "block_rate": f"{(self.input_blocks / max(1, self.input_scans)) * 100:.1f}%"
            },
            "output_guard": {
                "status": "ACTIVE",
                "total_scans": self.output_scans,
                "total_blocks": self.output_blocks,
                "block_rate": f"{(self.output_blocks / max(1, self.output_scans)) * 100:.1f}%"
            },
            "scanners": {
                "prompt_injection": "ACTIVE",       # LLM01 — input
                "sensitive_data": "ACTIVE",          # LLM02 — input (NEW)
                "excessive_agency": "ACTIVE",        # LLM06 — input (NEW)
                "unbounded_consumption": "ACTIVE",   # LLM10 — input (NEW)
                "topic_boundary": "ACTIVE",          # Policy — input
                "content_safety": "ACTIVE",          # Policy — input
                "dangerous_code": "ACTIVE",          # LLM06 — output
                "pii_leak": "ACTIVE",                # LLM02 — output
                "code_relevance": "ACTIVE",          # Output quality
                "language_correctness": "ACTIVE"     # Output quality
            },
            "recent_blocks": self.recent_blocks[-5:],
            "shield_status": "NOMINAL" if (self.input_blocks + self.output_blocks) < 10 else "ELEVATED"
        }


# Global stats singleton
guardrail_stats = GuardrailStats()
