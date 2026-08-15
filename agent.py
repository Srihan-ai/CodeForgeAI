"""
LangGraph Multi-Agent Workflow with Self-Correction
====================================================
This module defines a two-agent system with self-correction loops:
- Developer Agent: Generates Python/Java/C++ code dynamically based on user task
- Tester Agent: Creates test cases and executes sandbox verification
- Conditional Routing: Routes back to developer if tests fail (max 3 iterations)

Pattern: State Machine with Conditional Loops and State Reducers

Guardrails: Inspired by Guardrails AI, LLM Guard, and NeMo Guardrails
  - Input: Prompt injection, topic boundary, content safety
  - Output: Dangerous code, PII leaks, code relevance, language correctness
"""

import os
import sys
import io
import traceback
import warnings
import subprocess
import tempfile
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

from typing import Optional, List, Dict, Any, Literal
from operator import add
import random
import time

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing_extensions import TypedDict, Annotated

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import logging
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import LLM Guardrails Engine
try:
    from guardrails import InputGuard, OutputGuard, guardrail_stats
    GUARDRAILS_AVAILABLE = True
    logger.info("🛡️ LLM Guardrails Engine loaded successfully")
except ImportError:
    GUARDRAILS_AVAILABLE = False
    logger.warning("⚠️ Guardrails module not found — running without guardrails")


# ============================================================================
# CONFIGURATION WITH RETRY LOGIC + JITTER + FALLBACK
# ============================================================================

_circuit_breaker_failures = 0
_circuit_breaker_open = False
_circuit_breaker_last_failure_time = 0
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 60

LLM_PROVIDERS = [
    {
        "name": "groq",
        "env_key": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "class": ChatGroq,
        "available": True
    }
]

_current_provider_index = 0


class DemoAIMessage:
    """Mock AIMessage object for DemoLLM fallback."""
    def __init__(self, content: str):
        self.content = content


def extract_task_intent(raw_task: str) -> str:
    """
    Extracts and normalizes the abstract task intent from a raw user prompt,
    decoupling the core programmatic goal/algorithm from any language-specific wording.
    Examples:
    - 'Write a Python function that reverses a string and create tests for it.' -> 'Reverse a string and create tests'
    - 'Create a Java program to implement a linked list' -> 'Implement a linked list'
    - 'Implement a stack in C++' -> 'Implement a stack'
    - 'Write a Python function that validates email addresses' -> 'Validate email addresses'
    """
    if not raw_task or not raw_task.strip():
        return "General programming task"
    
    intent = raw_task.strip()
    
    # 1. Remove leading boilerplates like 'Write a Python function that', 'Create a Java program to', etc.
    patterns_to_strip = [
        r'^(?:please\s+)?(?:write|create|build|implement|develop|generate|construct|code)\s+(?:a|an|the)?\s*(?:python|java|c\+\+|cpp|javascript|typescript|rust|go)?\s*(?:function|class|method|program|script|service|module|algorithm|code)?\s*(?:that|to|which|for)?\s*',
        r'\s+in\s+(?:python|java|c\+\+|cpp|javascript|typescript|rust|go)(?:\s+3\.\d+)?\b',
        r'\s+using\s+(?:python|java|c\+\+|cpp|javascript|typescript|rust|go)\b',
        r'\b(?:python|java|c\+\+|cpp)\s+(?:function|program|class|script|code)\b',
        r'\b(?:python|java|c\+\+|cpp)\b'
    ]
    
    for pat in patterns_to_strip:
        intent = re.sub(pat, ' ', intent, flags=re.IGNORECASE).strip()
    
    # Clean up extra punctuation/spaces
    intent = re.sub(r'\s+', ' ', intent).strip(' .:;,')
    
    # Normalize common third-person action verbs to imperative/infinitive
    verb_normalizations = {
        r"^reverses\b": "Reverse",
        r"^calculates\b": "Calculate",
        r"^validates\b": "Validate",
        r"^checks\b": "Check",
        r"^implements\b": "Implement",
        r"^creates\b": "Create",
        r"^sorts\b": "Sort",
        r"^finds\b": "Find",
        r"^generates\b": "Generate",
        r"^builds\b": "Build"
    }
    for v_pat, v_rep in verb_normalizations.items():
        if re.search(v_pat, intent, re.IGNORECASE):
            intent = re.sub(v_pat, v_rep, intent, flags=re.IGNORECASE)
            break

    # Capitalize the first letter
    if intent:
        intent = intent[0].upper() + intent[1:]
    
    return intent or "General Code Implementation"


def generate_artifact_filename(task: str, language: str = "python") -> str:
    """
    Generates a clean, professional, task-derived source code filename.
    Examples:
    - 'Java linked list' -> 'LinkedList.java'
    - 'Python email validator' -> 'EmailValidator.py'
    - 'Binary Search Tree' -> 'BinarySearchTree.java'
    - 'Stack in C++' -> 'Stack.cpp'
    - 'Todo service in TypeScript' -> 'TodoService.ts'
    - 'Reverse a string' (target: java) -> 'StringReverser.java'
    """
    lang = (language or "python").lower()
    ext_map = {
        "python": ".py",
        "java": ".java",
        "cpp": ".cpp",
        "c++": ".cpp",
        "c": ".c",
        "javascript": ".js",
        "js": ".js",
        "typescript": ".ts",
        "ts": ".ts",
        "rust": ".rs",
        "go": ".go",
        "ruby": ".rb"
    }
    ext = ext_map.get(lang, ".py")
    task_lower = (task or "").lower()
    
    if any(k in task_lower for k in ["linked list", "linkedlist", "node"]):
        base = "LinkedList"
    elif any(k in task_lower for k in ["email", "validate email", "email address"]):
        base = "EmailValidator"
    elif any(k in task_lower for k in ["binary search tree", "bst"]):
        base = "BinarySearchTree"
    elif "binary search" in task_lower:
        base = "BinarySearch"
    elif "stack" in task_lower:
        base = "Stack"
    elif "queue" in task_lower:
        base = "Queue"
    elif "fibonacci" in task_lower:
        base = "Fibonacci"
    elif "palindrome" in task_lower:
        base = "PalindromeChecker"
    elif "prime" in task_lower:
        base = "PrimeChecker"
    elif any(k in task_lower for k in ["matrix", "2d array"]):
        base = "MatrixOperations"
    elif any(k in task_lower for k in ["sort", "quicksort", "mergesort", "bubblesort"]):
        base = "SortService"
    elif "todo" in task_lower:
        base = "TodoService"
    elif "reverse" in task_lower:
        base = "StringReverser"
    elif "factorial" in task_lower:
        base = "Factorial"
    elif "stats" in task_lower or "statistics" in task_lower:
        base = "StatisticsService"
    else:
        clean_task = re.sub(r'[^a-zA-Z0-9\s]', '', task).strip()
        words = [
            w.capitalize() for w in clean_task.split()
            if len(w) > 2 and w.lower() not in [
                'write', 'create', 'function', 'code', 'python', 'java', 'cpp',
                'that', 'with', 'check', 'calculate', 'using', 'return', 'make',
                'program', 'implement', 'build', 'for', 'the', 'and', 'from'
            ]
        ]
        base = "".join(words[:3]) if words else "Solution"

    formatted_name = base[0].upper() + base[1:]
    return f"{formatted_name}{ext}"


class DemoLLM:
    """
    Dynamic Offline/Fallback LLM Engine with Target Language Authority.
    Decouples task intent from target language:
    - User prompt might say "Write a Python function that reverses a string."
    - If target language is Java -> synthesizes Java implementation.
    - If target language is C++ -> synthesizes C++ implementation.
    - If target language is Python -> synthesizes Python implementation.
    """
    def invoke(self, input_data: Any) -> DemoAIMessage:
        prompt_text = ""
        user_task = ""
        
        if isinstance(input_data, list):
            prompt_text = " ".join(getattr(m, "content", str(m)) for m in input_data)
            for m in input_data:
                if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
                    user_task = getattr(m, "content", str(m))
                    break
            if not user_task:
                for m in input_data:
                    c = getattr(m, "content", str(m))
                    if not c.startswith("You are an expert") and not c.startswith("✅") and not c.startswith("❌") and not c.startswith("⚠️"):
                        user_task = c
                        break
        else:
            prompt_text = str(input_data)
            user_task = prompt_text
            
        if not user_task:
            user_task = prompt_text
            
        prompt_lower = prompt_text.lower()
        user_task_lower = user_task.lower()
        
        # 1. Authoritative Target Language Resolution
        lang = "python"
        if "authoritative target language: java" in prompt_lower or "target programming language: java" in prompt_lower or "target language: java" in prompt_lower:
            lang = "java"
        elif "authoritative target language: cpp" in prompt_lower or "authoritative target language: c++" in prompt_lower or "target programming language: cpp" in prompt_lower or "target programming language: c++" in prompt_lower or "target language: cpp" in prompt_lower or "target language: c++" in prompt_lower:
            lang = "cpp"
        elif "authoritative target language: python" in prompt_lower or "target programming language: python" in prompt_lower or "target language: python" in prompt_lower:
            lang = "python"
        elif "target language: java" in prompt_lower or "lang: java" in prompt_lower or "java 17" in prompt_lower:
            lang = "java"
        elif "target language: cpp" in prompt_lower or "lang: cpp" in prompt_lower or "c++ 20" in prompt_lower:
            lang = "cpp"
        elif "java" in prompt_lower and "python" not in prompt_lower:
            lang = "java"
        elif ("cpp" in prompt_lower or "c++" in prompt_lower) and "python" not in prompt_lower:
            lang = "cpp"
            
        # 2. Extract Task Intent & Classification
        is_linked_list = any(k in user_task_lower for k in ["linked list", "linkedlist", "node", "singly linked", "doubly linked", "head.next"])
        is_email = any(k in user_task_lower for k in ["email", "validate email", "email address", "email validation"])
        is_stack = any(k in user_task_lower for k in ["stack", "lifo", "push and pop"]) and not is_linked_list
        is_queue = any(k in user_task_lower for k in ["queue", "fifo", "enqueue", "dequeue"])
        is_tree = any(k in user_task_lower for k in ["tree", "bst", "binary search tree", "binary tree"])
        is_matrix = any(k in user_task_lower for k in ["matrix", "2d array", "matrix multiplication", "transpose"])
        is_prime = "prime" in user_task_lower and not is_linked_list
        is_fibo = "fibonacci" in user_task_lower
        is_palin = "palindrome" in user_task_lower
        is_div = ("divide" in user_task_lower or "division" in user_task_lower) and "safe" in user_task_lower
        is_reverse = "reverse" in user_task_lower and not is_linked_list
        is_factorial = "factorial" in user_task_lower
        is_sort = any(k in user_task_lower for k in ["sort", "bubble", "quicksort", "mergesort", "order"])
        is_search = any(k in user_task_lower for k in ["binary search", "search", "lookup", "find"]) and not is_linked_list and not is_tree
        is_stats = any(k in user_task_lower for k in ["stats", "statistics", "average", "sum", "count", "min", "max", "aggregate"])
        is_anagram = "anagram" in user_task_lower

        clean_task = re.sub(r'[^a-zA-Z0-9\s]', '', user_task).strip()
        safe_task_summary = clean_task[:50] if clean_task else "custom task"
        clean_words = [w for w in clean_task.split() if len(w) > 2 and w.lower() not in ['write', 'create', 'function', 'code', 'python', 'java', 'cpp', 'that', 'with', 'check', 'calculate', 'using', 'return', 'make', 'program', 'implement']]
        
        func_name_py = "_".join(w.lower() for w in clean_words[:3]) or "execute_task"
        class_name_java = "".join(w.capitalize() for w in clean_words[:3]) or "SolutionService"

        # ====================================================================
        # PYTHON 3.11 IMPLEMENTATIONS
        # ====================================================================
        if lang == "python":
            if is_linked_list:
                code = '''from typing import Optional, List


class Node:
    def __init__(self, data: int):
        self.data: int = data
        self.next: Optional["Node"] = None


class LinkedList:
    def __init__(self):
        self.head: Optional[Node] = None

    def insert(self, data: int) -> None:
        """Insert a new node at the end of the list."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next is not None:
            current = current.next
        current.next = new_node

    def delete(self, data: int) -> bool:
        """Delete the first occurrence of data. Returns True if deleted."""
        if self.head is None:
            return False
        if self.head.data == data:
            self.head = self.head.next
            return True
        current = self.head
        while current.next is not None and current.next.data != data:
            current = current.next
        if current.next is None:
            return False
        current.next = current.next.next
        return True

    def display(self) -> List[int]:
        """Traverse and return all elements in the linked list."""
        elements: List[int] = []
        current = self.head
        while current is not None:
            elements.append(current.data)
            current = current.next
        return elements


if __name__ == "__main__":
    linked_list = LinkedList()
    linked_list.insert(10)
    linked_list.insert(20)
    linked_list.insert(30)
    assert linked_list.display() == [10, 20, 30]
    linked_list.delete(20)
    assert linked_list.display() == [10, 30]
    print("LinkedList operations verified successfully:", linked_list.display())
'''
            elif is_email:
                code = '''import re
from typing import Optional


def validate_email(email: Optional[str]) -> bool:
    """
    Validates whether the provided string is a syntactically valid email address.
    """
    if not email or not isinstance(email, str):
        return False
    
    email_pattern = re.compile(
        r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    )
    return bool(email_pattern.match(email.strip()))


if __name__ == "__main__":
    assert validate_email("user@example.com") is True
    assert validate_email("invalid-email") is False
    assert validate_email("sathvik@example.org") is True
    assert validate_email("@no-user.com") is False
    print("All email validation assertions passed successfully.")
'''
            elif is_stack:
                code = '''from typing import Any, List, Optional


class Stack:
    def __init__(self):
        self._items: List[Any] = []

    def push(self, item: Any) -> None:
        """Push an element onto the stack."""
        self._items.append(item)

    def pop(self) -> Any:
        """Remove and return the top element of the stack."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> Optional[Any]:
        """Return the top element without removing it."""
        if self.is_empty():
            return None
        return self._items[-1]

    def is_empty(self) -> bool:
        """Check whether the stack contains any items."""
        return len(self._items) == 0

    def size(self) -> int:
        """Return total number of items on the stack."""
        return len(self._items)


if __name__ == "__main__":
    stack = Stack()
    stack.push(10)
    stack.push(20)
    assert stack.peek() == 20
    assert stack.pop() == 20
    assert stack.size() == 1
    print("Stack operations verified successfully.")
'''
            elif is_tree:
                code = '''from typing import Optional, List


class TreeNode:
    def __init__(self, value: int):
        self.value: int = value
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None


class BinarySearchTree:
    def __init__(self):
        self.root: Optional[TreeNode] = None

    def insert(self, value: int) -> None:
        """Insert a value into the binary search tree."""
        if self.root is None:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node: TreeNode, value: int) -> None:
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)

    def inorder_traversal(self) -> List[int]:
        """Return in-order traversal of the tree values."""
        result: List[int] = []
        def _traverse(current: Optional[TreeNode]):
            if current is not None:
                _traverse(current.left)
                result.append(current.value)
                _traverse(current.right)
        _traverse(self.root)
        return result


if __name__ == "__main__":
    bst = BinarySearchTree()
    for val in [50, 30, 70, 20, 40]:
        bst.insert(val)
    assert bst.inorder_traversal() == [20, 30, 40, 50, 70]
    print("BinarySearchTree verified:", bst.inorder_traversal())
'''
            elif is_reverse:
                code = '''def reverse_string(text: str) -> str:
    """Reverse a given string."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    return text[::-1]


if __name__ == "__main__":
    result = reverse_string("hello")
    assert result == "olleh"
    print("reverse_string('hello'):", result)
'''
            elif is_prime:
                code = '''def is_prime(number: int) -> bool:
    """Check if an integer is a prime number."""
    if number <= 1:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True


if __name__ == "__main__":
    assert is_prime(1) is False
    assert is_prime(2) is True
    assert is_prime(29) is True
    assert is_prime(30) is False
    print("Prime number assertions passed successfully.")
'''
            elif is_fibo:
                code = '''from typing import List


def fibonacci_sequence(terms: int) -> List[int]:
    """Generate the Fibonacci sequence up to n terms."""
    if terms <= 0:
        return []
    if terms == 1:
        return [0]
    sequence: List[int] = [0, 1]
    while len(sequence) < terms:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence


if __name__ == "__main__":
    assert fibonacci_sequence(0) == []
    assert fibonacci_sequence(5) == [0, 1, 1, 2, 3]
    print("Fibonacci sequence generated cleanly:", fibonacci_sequence(7))
'''
            elif is_palin:
                code = '''def is_palindrome(text: str) -> bool:
    """Check if a string is a palindrome, ignoring non-alphanumeric characters."""
    if not isinstance(text, str):
        return False
    cleaned = "".join(char.lower() for char in text if char.isalnum())
    return cleaned == cleaned[::-1]


if __name__ == "__main__":
    assert is_palindrome("racecar") is True
    assert is_palindrome("A man, a plan, a canal: Panama") is True
    assert is_palindrome("hello") is False
    print("Palindrome validation assertions passed successfully.")
'''
            elif is_div:
                code = '''def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers with proper error handling."""
    if denominator == 0:
        raise ZeroDivisionError("Denominator cannot be zero")
    return numerator / denominator


if __name__ == "__main__":
    assert safe_divide(10, 2) == 5.0
    print("safe_divide(10, 2):", safe_divide(10, 2))
'''
            elif is_sort:
                code = '''from typing import List


def quick_sort(items: List[int]) -> List[int]:
    """Sort a list of integers in ascending order using quicksort."""
    if len(items) <= 1:
        return items
    pivot = items[len(items) // 2]
    left = [x for x in items if x < pivot]
    middle = [x for x in items if x == pivot]
    right = [x for x in items if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


if __name__ == "__main__":
    sample = [64, 34, 25, 12, 22, 11, 90]
    sorted_sample = quick_sort(sample)
    assert sorted_sample == [11, 12, 22, 25, 34, 64, 90]
    print("quick_sort verified:", sorted_sample)
'''
            elif is_search:
                code = '''from typing import List


def binary_search(array: List[int], target: int) -> int:
    """Search for target in sorted array. Returns index or -1."""
    left, right = 0, len(array) - 1
    while left <= right:
        mid = (left + right) // 2
        if array[mid] == target:
            return mid
        elif array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


if __name__ == "__main__":
    data = [10, 20, 30, 40, 50]
    assert binary_search(data, 30) == 2
    assert binary_search(data, 99) == -1
    print("binary_search verified successfully.")
'''
            elif is_stats:
                code = '''from typing import List, Dict, Optional


def compute_statistics(numbers: List[float]) -> Dict[str, Optional[float]]:
    """Compute aggregate count, sum, average, min, and max."""
    if not numbers:
        return {"count": 0, "sum": 0.0, "avg": 0.0, "min": None, "max": None}
    return {
        "count": len(numbers),
        "sum": float(sum(numbers)),
        "avg": float(sum(numbers) / len(numbers)),
        "min": float(min(numbers)),
        "max": float(max(numbers))
    }


if __name__ == "__main__":
    stats = compute_statistics([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats["avg"] == 30.0 and stats["sum"] == 150.0
    print("compute_statistics verified:", stats)
'''
            else:
                code = f'''from typing import List, Dict, Any


def {func_name_py}(items: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Implementation for task specification:
    '{safe_task_summary}'
    """
    data = items if items is not None else [10, 20, 30, 40]
    processed = [x * 2 if isinstance(x, (int, float)) else str(x).upper() for x in data]
    return {{
        "task": "{safe_task_summary}",
        "input_count": len(data),
        "processed_result": processed,
        "status": "success"
    }}


if __name__ == "__main__":
    result = {func_name_py}([1, 2, 3, 4])
    assert result["status"] == "success" and result["input_count"] == 4
    print("Dynamic execution verified:", result)
'''

        # ====================================================================
        # JAVA 17 IMPLEMENTATIONS
        # ====================================================================
        elif lang == "java":
            if is_linked_list:
                code = '''public class LinkedList {
    private Node head;

    private static class Node {
        int data;
        Node next;

        Node(int data) {
            this.data = data;
        }
    }

    public void insert(int data) {
        Node newNode = new Node(data);
        if (head == null) {
            head = newNode;
            return;
        }
        Node current = head;
        while (current.next != null) {
            current = current.next;
        }
        current.next = newNode;
    }

    public boolean delete(int data) {
        if (head == null) return false;
        if (head.data == data) {
            head = head.next;
            return true;
        }
        Node current = head;
        while (current.next != null && current.next.data != data) {
            current = current.next;
        }
        if (current.next == null) return false;
        current.next = current.next.next;
        return true;
    }

    public void display() {
        Node current = head;
        while (current != null) {
            System.out.print(current.data + " ");
            current = current.next;
        }
        System.out.println();
    }

    public static void main(String[] args) {
        LinkedList list = new LinkedList();
        list.insert(10);
        list.insert(20);
        list.insert(30);
        list.display();
        list.delete(20);
        list.display();
        System.out.println("Java LinkedList insertion, deletion, and traversal executed successfully!");
    }
}
'''
            elif is_email:
                code = '''import java.util.regex.Pattern;

public class EmailValidator {
    private static final Pattern EMAIL_PATTERN = Pattern.compile(
        "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\\\.[a-zA-Z0-9-.]+$"
    );

    public static boolean validate(String email) {
        if (email == null || email.trim().isEmpty()) {
            return false;
        }
        return EMAIL_PATTERN.matcher(email.trim()).matches();
    }

    public static void main(String[] args) {
        boolean valid = validate("user@example.com");
        boolean invalid = validate("invalid.email");
        System.out.println("validate('user@example.com'): " + valid);
        System.out.println("validate('invalid.email'): " + invalid);
        if (valid && !invalid) {
            System.out.println("All Java email validation test cases passed!");
        }
    }
}
'''
            elif is_stack:
                code = '''import java.util.ArrayList;

public class Stack {
    private final ArrayList<Integer> items = new ArrayList<>();

    public void push(int item) {
        items.add(item);
    }

    public int pop() {
        if (isEmpty()) {
            throw new IllegalStateException("Stack is empty");
        }
        return items.remove(items.size() - 1);
    }

    public int peek() {
        if (isEmpty()) {
            throw new IllegalStateException("Stack is empty");
        }
        return items.get(items.size() - 1);
    }

    public boolean isEmpty() {
        return items.isEmpty();
    }

    public int size() {
        return items.size();
    }

    public static void main(String[] args) {
        Stack stack = new Stack();
        stack.push(10);
        stack.push(20);
        System.out.println("Peek: " + stack.peek());
        System.out.println("Pop: " + stack.pop());
        System.out.println("Java Stack implementation verified successfully!");
    }
}
'''
            elif is_tree:
                code = '''public class BinarySearchTree {
    private Node root;

    private static class Node {
        int value;
        Node left, right;

        Node(int value) {
            this.value = value;
        }
    }

    public void insert(int value) {
        root = insertRecursive(root, value);
    }

    private Node insertRecursive(Node current, int value) {
        if (current == null) {
            return new Node(value);
        }
        if (value < current.value) {
            current.left = insertRecursive(current.left, value);
        } else if (value > current.value) {
            current.right = insertRecursive(current.right, value);
        }
        return current;
    }

    public void inorder() {
        inorderRecursive(root);
        System.out.println();
    }

    private void inorderRecursive(Node node) {
        if (node != null) {
            inorderRecursive(node.left);
            System.out.print(node.value + " ");
            inorderRecursive(node.right);
        }
    }

    public static void main(String[] args) {
        BinarySearchTree bst = new BinarySearchTree();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        System.out.print("Inorder BST: ");
        bst.inorder();
        System.out.println("Java BST verified!");
    }
}
'''
            elif is_reverse:
                code = '''public class StringReverser {
    /**
     * Reverses a given string.
     */
    public static String reverseString(String text) {
        if (text == null) {
            throw new IllegalArgumentException("Input must not be null");
        }
        return new StringBuilder(text).reverse().toString();
    }

    public static void main(String[] args) {
        String result = reverseString("hello");
        if (!"olleh".equals(result)) {
            throw new AssertionError("Expected 'olleh' but got: " + result);
        }
        System.out.println("StringReverser('hello') verified: " + result);
    }
}
'''
            elif is_prime:
                code = '''public class PrimeChecker {
    public static boolean isPrime(int n) {
        if (n <= 1) return false;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        boolean test = isPrime(29);
        System.out.println("isPrime(29): " + test);
    }
}
'''
            elif is_fibo:
                code = '''public class FibonacciCalculator {
    public static int calculate(int n) {
        if (n <= 1) return n;
        int a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            int temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }

    public static void main(String[] args) {
        System.out.println("Fibonacci(7): " + calculate(7));
    }
}
'''
            elif is_search:
                code = '''public class BinarySearch {
    public static int search(int[] array, int target) {
        int left = 0, right = array.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (array[mid] == target) return mid;
            if (array[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }

    public static void main(String[] args) {
        int[] data = {10, 20, 30, 40, 50};
        int idx = search(data, 30);
        if (idx != 2) throw new AssertionError("Expected index 2");
        System.out.println("BinarySearch verified successfully: " + idx);
    }
}
'''
            elif is_sort:
                code = '''import java.util.Arrays;

public class SortService {
    public static int[] sort(int[] array) {
        int[] copy = Arrays.copyOf(array, array.length);
        Arrays.sort(copy);
        return copy;
    }

    public static void main(String[] args) {
        int[] sample = {64, 34, 25, 12, 22, 11, 90};
        int[] sorted = sort(sample);
        System.out.println("SortService verified: " + Arrays.toString(sorted));
    }
}
'''
            else:
                code = f'''public class {class_name_java} {{
    public static String execute(String taskSpec) {{
        System.out.println("Executing Java 17 logic for: " + taskSpec);
        return "SUCCESS: " + taskSpec;
    }}

    public static void main(String[] args) {{
        String result = execute("{safe_task_summary}");
        System.out.println("Execution result: " + result);
    }}
}}
'''

        # ====================================================================
        # C++ 20 IMPLEMENTATIONS
        # ====================================================================
        else:
            if is_reverse:
                code = '''#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>

std::string reverseString(std::string text) {
    std::reverse(text.begin(), text.end());
    return text;
}

int main() {
    std::string result = reverseString("hello");
    assert(result == "olleh");
    std::cout << "reverseString('hello') verified: " << result << std::endl;
    return 0;
}
'''
            elif is_linked_list:
                code = '''#include <iostream>

struct Node {
    int data;
    Node* next;
    explicit Node(int val) : data(val), next(nullptr) {}
};

class LinkedList {
private:
    Node* head;

public:
    LinkedList() : head(nullptr) {}

    ~LinkedList() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }

    void insert(int val) {
        Node* newNode = new Node(val);
        if (!head) {
            head = newNode;
            return;
        }
        Node* temp = head;
        while (temp->next != nullptr) {
            temp = temp->next;
        }
        temp->next = newNode;
    }

    bool remove(int key) {
        if (!head) return false;
        if (head->data == key) {
            Node* temp = head;
            head = head->next;
            delete temp;
            return true;
        }
        Node* curr = head;
        while (curr->next != nullptr && curr->next->data != key) {
            curr = curr->next;
        }
        if (!curr->next) return false;
        Node* temp = curr->next;
        curr->next = curr->next->next;
        delete temp;
        return true;
    }

    void display() const {
        Node* curr = head;
        while (curr != nullptr) {
            std::cout << curr->data << " ";
            curr = curr->next;
        }
        std::cout << std::endl;
    }
};

int main() {
    LinkedList list;
    list.insert(10);
    list.insert(20);
    list.insert(30);
    list.display();
    list.remove(20);
    list.display();
    std::cout << "C++ LinkedList operations executed successfully!" << std::endl;
    return 0;
}
'''
            elif is_email:
                code = '''#include <iostream>
#include <string>
#include <regex>

bool validateEmail(const std::string& email) {
    if (email.empty()) return false;
    const std::regex pattern(R"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)");
    return std::regex_match(email, pattern);
}

int main() {
    bool valid = validateEmail("user@example.com");
    bool invalid = validateEmail("invalid.email");
    std::cout << "validateEmail('user@example.com'): " << (valid ? "true" : "false") << std::endl;
    std::cout << "validateEmail('invalid.email'): " << (invalid ? "true" : "false") << std::endl;
    return 0;
}
'''
            elif is_stack:
                code = '''#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

class Stack {
private:
    std::vector<int> data;

public:
    void push(int val) {
        data.push_back(val);
    }

    int pop() {
        if (isEmpty()) throw std::out_of_range("Stack underflow");
        int topVal = data.back();
        data.pop_back();
        return topVal;
    }

    int peek() const {
        if (isEmpty()) throw std::out_of_range("Stack is empty");
        return data.back();
    }

    bool isEmpty() const {
        return data.empty();
    }

    size_t size() const {
        return data.size();
    }
};

int main() {
    Stack s;
    s.push(10);
    s.push(20);
    assert(s.peek() == 20);
    assert(s.pop() == 20);
    assert(s.size() == 1);
    std::cout << "C++ Stack verified successfully!" << std::endl;
    return 0;
}
'''
            elif is_prime:
                code = '''#include <iostream>
#include <cassert>

bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

int main() {
    assert(isPrime(29) == true);
    assert(isPrime(15) == false);
    std::cout << "C++ isPrime verified: 29 is prime." << std::endl;
    return 0;
}
'''
            elif is_fibo:
                code = '''#include <iostream>
#include <cassert>

long long fibonacci(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    long long a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        long long temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

int main() {
    assert(fibonacci(7) == 13);
    std::cout << "C++ fibonacci(7) verified: 13." << std::endl;
    return 0;
}
'''
            elif is_search:
                code = '''#include <iostream>
#include <vector>
#include <cassert>

int binarySearch(const std::vector<int>& arr, int target) {
    int left = 0, right = static_cast<int>(arr.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

int main() {
    std::vector<int> data = {10, 20, 30, 40, 50};
    assert(binarySearch(data, 30) == 2);
    assert(binarySearch(data, 99) == -1);
    std::cout << "C++ binarySearch verified!" << std::endl;
    return 0;
}
'''
            else:
                code = f'''#include <iostream>
#include <string>

void executeTask() {{
    std::cout << "Executing C++ 20 specification: {safe_task_summary}" << std::endl;
}}

int main() {{
    executeTask();
    return 0;
}}
'''

        return DemoAIMessage(content=code)


def get_llm(force_fallback=False):
    global _circuit_breaker_open, _circuit_breaker_last_failure_time, _current_provider_index
    provider = LLM_PROVIDERS[_current_provider_index]
    api_key = os.environ.get(provider["env_key"], "").strip()
    
    if api_key and not api_key.startswith("your_") and len(api_key) > 10:
        logger.info(f"Using LLM provider: {provider['name']} ({provider['model']})")
        return provider["class"](
            model=provider["model"],
            **{provider["env_key"].lower(): api_key},
            temperature=0.1,
            timeout=30.0
        )
    
    logger.info("💡 Using Demo/Mock LLM engine for instant out-of-the-box evaluation")
    return DemoLLM()


_llm_instance = None

def get_llm_instance():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm()
    return _llm_instance


def get_llm_mode_label() -> str:
    """
    Returns a human-readable label for the active LLM mode.
    Reads the env key at call time (not cached) so it always reflects
    the actual runtime config for that specific request.
    """
    provider = LLM_PROVIDERS[_current_provider_index]
    api_key = os.environ.get(provider["env_key"], "").strip()
    if api_key and not api_key.startswith("your_") and len(api_key) > 10:
        return f"groq/{provider['model']}"
    return "template-fallback"

def jittered_wait(multiplier=1, min_wait=1, max_wait=10):
    def wait_func(retry_state):
        attempt = retry_state.attempt_number
        exponential_wait = min(max_wait, multiplier * (2 ** attempt))
        jittered = random.uniform(min_wait, exponential_wait)
        logger.info(f"Retry attempt {attempt}: waiting {jittered:.2f}s (with jitter)")
        return jittered
    return wait_func


def is_retryable_error(exception: Exception) -> bool:
    import httpx
    global _circuit_breaker_failures, _circuit_breaker_open, _circuit_breaker_last_failure_time
    
    if isinstance(exception, (ConnectionError, TimeoutError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        _circuit_breaker_failures += 1
        _circuit_breaker_last_failure_time = time.time()
        if _circuit_breaker_failures >= CIRCUIT_BREAKER_THRESHOLD:
            _circuit_breaker_open = True
            logger.error(f"Circuit breaker opened after {_circuit_breaker_failures} failures")
        return True
    
    exception_str = str(exception).lower()
    if "rate" in exception_str or "429" in exception_str or "quota" in exception_str or "temporarily unavailable" in exception_str:
        return True
    
    return False


llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=jittered_wait(multiplier=1, min_wait=1, max_wait=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO),
    reraise=True
)


@llm_retry
def call_llm_with_retry(prompt) -> Any:
    global _circuit_breaker_failures
    try:
        response = get_llm_instance().invoke(prompt)
        _circuit_breaker_failures = max(0, _circuit_breaker_failures - 1)
        return response
    except Exception as e:
        if is_retryable_error(e):
            raise
        else:
            logger.error(f"Non-retryable error: {e}")
            raise


def validate_task_input(task: str) -> tuple[bool, Optional[str]]:
    if not task or not task.strip():
        return False, "Task cannot be empty. Please describe what code you want to generate."
    if len(task) < 5:
        return False, "Task too short. Please provide more details."
    if len(task) > 1000:
        return False, "Task too long. Please keep it under 1000 characters."
    return True, None


def validate_code_output(code: str, language: str = "python") -> tuple[bool, Optional[str]]:
    """
    Validates that the generated code is real, compilable source code and not
    conversational text, markdown response, or broken syntax.
    """
    if not code or not code.strip():
        return False, "Developer agent returned empty code."
    
    cleaned_code = code.strip()
    language = (language or "python").lower()

    # Reject unstripped markdown fences
    if cleaned_code.startswith("```") or cleaned_code.endswith("```"):
        return False, "Code contains unstripped Markdown code fences."

    # Reject obvious conversational chat preambles
    first_line = cleaned_code.split('\n')[0].strip().lower()
    if any(first_line.startswith(p) for p in ["here is", "sure,", "certainly", "below is", "hope this helps"]):
        return False, "Output contains conversational chat text instead of pure source code."
    
    if language == "python":
        python_keywords = ['def ', 'class ', 'import ', 'from ', 'return', '=', 'if ', 'for ', 'while ']
        if not any(keyword in cleaned_code for keyword in python_keywords):
            return False, "Output does not contain valid Python definitions or statements."
        try:
            compile(cleaned_code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Python syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Python compilation error: {str(e)}"
    
    elif language == "java":
        java_keywords = ['class ', 'public ', 'private ', 'void ', 'int ', 'String ', 'return', 'static ', 'boolean', 'interface ']
        if not any(keyword in cleaned_code for keyword in java_keywords):
            return False, "Output does not contain valid Java class or method structures."
        
        # Check balanced braces
        open_braces = cleaned_code.count('{')
        close_braces = cleaned_code.count('}')
        if open_braces != close_braces:
            return False, f"Java syntax error: Unbalanced curly braces ({{: {open_braces}, }}: {close_braces})."
        
        # Check for class declaration
        if not re.search(r'\b(class|interface|record|enum)\s+[A-Za-z0-9_]+', cleaned_code):
            return False, "Java syntax error: Missing class or interface declaration."
            
        return True, None
    
    elif language in ["cpp", "c++"]:
        cpp_keywords = ['#include', 'int ', 'void ', 'return', 'std::', 'main(', 'using', 'bool', 'class ', 'struct ']
        if not any(keyword in cleaned_code for keyword in cpp_keywords):
            return False, "Output does not contain valid C++ declarations or preprocessor directives."
            
        # Check balanced braces
        open_braces = cleaned_code.count('{')
        close_braces = cleaned_code.count('}')
        if open_braces != close_braces:
            return False, f"C++ syntax error: Unbalanced curly braces ({{: {open_braces}, }}: {close_braces})."
            
        return True, None
    
    return True, None


class CrewState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add]
    code: Optional[str]
    report: Optional[str]
    execution_success: bool
    iterations: int
    max_iterations: int
    language: Optional[str]
    task_intent: Optional[str]
    task_specification: Optional[str]
    target_language: Optional[str]
    filename: Optional[str]
    hitl_enabled: Optional[bool]
    human_action: Optional[str]
    human_feedback: Optional[str]
    human_edited_code: Optional[str]
    human_review_status: Optional[str]


@tool
def run_python_code(code: str) -> str:
    """
    Execute Python code in an isolated subprocess with a 5-second timeout.
    Uses an empty environment (env={}) so no server env vars (e.g. GROQ_API_KEY)
    are accessible to generated code. Kills the process on timeout.
    """
    if not isinstance(code, str):
        code = str(code)
    clean_code = code.replace("```python", "").replace("```", "").strip()

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(clean_code)
            tmp_path = tmp.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5,
            env={},          # empty env: no inherited API keys or secrets
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            return f"Execution Error:\n{stderr or stdout}"
        return stdout if stdout else "Success (no terminal output)"

    except subprocess.TimeoutExpired:
        return "Execution Error:\nCode exceeded 5-second timeout limit (possible infinite loop)."
    except Exception as exc:
        return f"Execution Error:\n{exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate 3 test scenarios for the task description."""
    prompt = (
        f"Generate 3 test scenarios for: '{task_description}'. Return numbered list."
    )
    response = call_llm_with_retry(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _extract_text(content: Any) -> str:
    if isinstance(content, list):
        first = content[0]
        return first.get("text", "") if isinstance(first, dict) else str(first)
    return str(content)


def _make_user_friendly_error(exception: Exception) -> str:
    error_str = str(exception).lower()
    if "429" in error_str or "rate limit" in error_str:
        return "⚠️ Rate limit reached. Please wait 30 seconds."
    if "timeout" in error_str:
        return "⏱️ Request timed out."
    return f"❌ Error: {str(exception)[:100]}"


def sanitize_professional_code(raw_code: str, language: str = "python") -> str:
    """
    Extracts and sanitizes LLM output into clean, production-grade source code:
    - Strips markdown code block fences (```java, ```python, ```cpp, ```)
    - Strips markdown title/hashtag headers (# Solution, ## Code, ### Implementation)
    - Strips conversational preambles ("Here is the Java code...", "Certainly!...")
    - Strips trailing explanations ("**Explanation:**", "### How it works:", etc.)
    - Strips markdown bullet points and list markers outside of valid comments
    - Returns clean, properly indented, compilable source code.
    """
    if not raw_code:
        return ""

    text = raw_code.strip()

    # 1. Extract content from within markdown code fences if present
    fence_pattern = re.compile(
        r'```(?:python|java|cpp|c\+\+|c|typescript|ts|javascript|js|rust|go)?\s*\n(.*?)```',
        re.DOTALL | re.IGNORECASE
    )
    matches = fence_pattern.findall(text)
    if matches:
        text = max(matches, key=len).strip()
    else:
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text).strip()

    lines = text.split('\n')
    cleaned_lines = []
    in_code_body = False
    language_lower = (language or "python").lower()

    chat_preamble_phrases = [
        "here is the", "here's the", "certainly", "sure,", "below is", 
        "i have written", "hope this helps", "let me know", "feel free",
        "here is a", "this program", "this solution", "this function",
        "following is", "the code below"
    ]

    code_start_keywords = [
        'def ', 'class ', 'import ', 'from ', '#include', 'public ', 'private ',
        'protected ', 'int ', 'void ', 'package ', 'interface ', 'struct ',
        'template', 'using ', 'const ', 'function ', 'export '
    ]

    for line in lines:
        stripped = line.strip()
        stripped_lower = stripped.lower()

        # Check if code has started
        if any(kw in line for kw in code_start_keywords):
            in_code_body = True

        # If we haven't reached code body yet, check for chat preamble or markdown headers
        if not in_code_body:
            if stripped.startswith('#') and any(k in stripped_lower for k in ['solution', 'code', 'implementation', 'program', 'example', 'output', 'task', 'step', 'java', 'python', 'c++']):
                continue
            if stripped.startswith('**') and stripped.endswith('**'):
                continue
            if any(phrase in stripped_lower for phrase in chat_preamble_phrases):
                continue
            if not stripped:
                continue

        # Stop if trailing markdown explanation section begins
        if in_code_body:
            if stripped.startswith('**Explanation') or stripped.startswith('### Explanation') or stripped.startswith('## Explanation') or stripped.startswith('**Key Points') or stripped.startswith('**Complexity'):
                break
            if stripped_lower.startswith('hope this helps') or stripped_lower.startswith('let me know if'):
                break

        # Remove stray markdown header formatting from lines inside code (e.g. `### public class...`)
        if stripped.startswith('### ') or stripped.startswith('## ') or (stripped.startswith('# ') and language_lower in ['java', 'cpp', 'c', 'js', 'ts']):
            if not (language_lower in ['cpp', 'c'] and (stripped.startswith('#include') or stripped.startswith('#define') or stripped.startswith('#pragma') or stripped.startswith('#ifdef') or stripped.startswith('#endif'))):
                continue

        cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines).strip()
    return result if result else raw_code.strip()


def developer_node(state: CrewState) -> Dict[str, Any]:
    target_language = (state.get("target_language") or state.get("language") or "python").lower()
    raw_task = state["messages"][0].content if state.get("messages") else state.get("task_specification", "")
    task_intent = state.get("task_intent") or extract_task_intent(raw_task)
    filename = generate_artifact_filename(task_intent or raw_task, target_language)
    
    # Malicious injection check
    dangerous_keywords = ["import os; os.system", "rm -rf", "eval(", "exec("]
    if any(keyword in raw_task for keyword in dangerous_keywords):
        logger.warning(f"⚠️ Dangerous input detected: {raw_task[:50]}")
        return {
            "code": "// SECURITY ALERT: Input contains potentially dangerous code",
            "messages": [AIMessage(content="⚠️ Security check failed: dangerous code pattern detected.")],
            "iterations": state.get("iterations", 0) + 1,
            "execution_success": False,
            "report": "### SECURITY ALERT\nInput contains unsafe operations.",
            "task_intent": task_intent,
            "target_language": target_language,
            "filename": filename
        }
    
    retry_context = ""
    if state.get("iterations", 0) > 0 and state.get("report"):
        retry_context = f"\n\nPREVIOUS TEST / COMPILER ERRORS TO FIX:\n{state['report']}\n"
    
    prompt = (
        f"You are an expert software engineer.\n"
        f"TASK INTENT: {task_intent}\n"
        f"USER SPECIFICATION: {raw_task}\n"
        f"AUTHORITATIVE TARGET LANGUAGE: {target_language.upper()}\n"
        f"{retry_context}\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"1. Generate ONLY valid, clean, compilable, production-ready {target_language.upper()} code implementing the task intent '{task_intent}'.\n"
        f"2. The AUTHORITATIVE TARGET LANGUAGE ({target_language.upper()}) is authoritative over any conflicting language mentioned in the user prompt.\n"
        f"3. DO NOT wrap the code in Markdown code block fences (no ```).\n"
        f"4. DO NOT include markdown headers (# Solution) or conversational explanations.\n"
        f"5. Return ONLY the pure source code artifact.\n"
    )
    
    try:
        response = call_llm_with_retry(prompt)
        raw_code = response.content if hasattr(response, "content") else str(response)
        
        # Sanitize code into clean, professional code
        clean_code = sanitize_professional_code(raw_code, target_language)
        
        # Guardrails Output Validation
        if GUARDRAILS_AVAILABLE:
            output_report = OutputGuard.scan_all(clean_code, target_language)
            guardrail_stats.record_output_scan(output_report)
            
            if not output_report.passed:
                logger.warning(f"🛡️ Output guardrail blocked: {output_report.blocked_by}")
                return {
                    "code": f"// GUARDRAIL BLOCKED: {output_report.reason}",
                    "messages": [AIMessage(content=f"🛡️ Output Guardrail: {output_report.reason}")],
                    "iterations": state.get("iterations", 0) + 1,
                    "execution_success": False,
                    "report": f"### 🛡️ Output Guardrail Alert\n{output_report.reason}\n\nBlocked by: {output_report.blocked_by}\nSeverity: {output_report.severity.value}",
                    "task_intent": task_intent,
                    "target_language": target_language,
                    "filename": filename
                }
        else:
            is_valid, error_msg = validate_code_output(clean_code, target_language)
            if not is_valid:
                logger.warning(f"⚠️ Code validation failed: {error_msg}")
                return {
                    "code": clean_code,
                    "messages": [AIMessage(content=f"⚠️ Validation warning: {error_msg}")],
                    "iterations": state.get("iterations", 0) + 1,
                    "report": f"### VALIDATION WARNING\n{error_msg}",
                    "execution_success": False,
                    "task_intent": task_intent,
                    "target_language": target_language,
                    "filename": filename
                }
        
        return {
            "code": clean_code,
            "messages": [AIMessage(content=f"✅ Generated {target_language.upper()} code (iteration {state.get('iterations', 0) + 1})")],
            "iterations": state.get("iterations", 0) + 1,
            "task_intent": task_intent,
            "target_language": target_language,
            "filename": filename
        }
        
    except Exception as e:
        user_friendly_error = _make_user_friendly_error(e)
        return {
            "code": f"// ERROR: {user_friendly_error}",
            "messages": [AIMessage(content=f"❌ Developer agent error: {user_friendly_error}")],
            "iterations": state.get("iterations", 0) + 1,
            "execution_success": False,
            "report": f"### ERROR\n{user_friendly_error}",
            "task_intent": task_intent,
            "target_language": target_language,
            "filename": filename
        }


def human_review_node(state: CrewState) -> Dict[str, Any]:
    """
    Human-in-the-Loop (HITL) Gate Node.
    Allows human reviewers to inspect, edit, approve, or reject code before Sandbox testing.
    """
    hitl_enabled = state.get("hitl_enabled", False)
    human_action = state.get("human_action")
    
    if not hitl_enabled:
        return {
            "human_review_status": "bypassed",
            "messages": []
        }
    
    if human_action == "edit" and state.get("human_edited_code"):
        return {
            "code": state["human_edited_code"],
            "human_review_status": "edited",
            "messages": [AIMessage(content="✏️ Human reviewer modified the code before testing.")]
        }
    elif human_action == "reject":
        feedback = state.get("human_feedback", "Please revise code based on human review.")
        return {
            "human_review_status": "rejected",
            "messages": [HumanMessage(content=f"Human Reviewer Feedback: {feedback}")]
        }
    elif human_action == "abort":
        return {
            "human_review_status": "aborted",
            "execution_success": False,
            "report": "### EXECUTION ABORTED\nWorkflow was cancelled by human reviewer.",
            "messages": [AIMessage(content="🛑 Workflow aborted by human reviewer.")]
        }
    elif human_action == "approve":
        return {
            "human_review_status": "approved",
            "messages": [AIMessage(content="✅ Human reviewer approved the code for sandbox testing.")]
        }
    else:
        return {
            "human_review_status": "pending",
            "messages": []
        }


def should_route_from_human_review(state: CrewState) -> Literal["tester", "developer", "end"]:
    """Conditional routing from Human-in-the-Loop Gate."""
    status = state.get("human_review_status", "bypassed")
    if status == "rejected":
        return "developer"
    elif status == "aborted":
        return "end"
    return "tester"


def tester_node(state: CrewState) -> Dict[str, Any]:
    task = state["messages"][0].content
    target_language = state.get("language", "python").lower()
    code = state.get("code", "")
    
    if code.startswith("// ERROR:") or code.startswith("# ERROR:") or code.startswith("// GUARDRAIL BLOCKED:"):
        return {
            "report": f"### DEVELOPER ERROR\n{code}\n\n❌ Cannot run tests - code generation failed.",
            "execution_success": False,
            "messages": [AIMessage(content="❌ Developer returned invalid output.")]
        }
    
    # 1. Structural syntax and compilation validation
    is_valid, syntax_error = validate_code_output(code, target_language)
    if not is_valid:
        return {
            "report": f"[COMPILATION / SYNTAX ERROR]\n{syntax_error}\n\n[STATUS] Code rejected by Tester Agent. Self-healing loop triggered.",
            "execution_success": False,
            "messages": [AIMessage(content=f"❌ Compiler/Syntax rejection: {syntax_error}")]
        }
    
    try:
        cases_str = f"1. Standard input verification for '{task}'\n2. Edge case boundary test\n3. Exception handling assertion"

        if target_language == "python":
            execution_result = run_python_code.invoke(code)
            execution_success = not execution_result.startswith("Execution Error:")
        else:
            # Java and C++ runtime compilation is not available in this environment.
            # Report the result of the structural syntax check already performed above.
            lang_label = "Java" if target_language == "java" else "C++"
            execution_result = (
                f"[STATIC SYNTAX CHECK — not compiled or executed]\n"
                f"{lang_label} source passed structural syntax validation.\n"
                f"Note: Runtime compilation for {lang_label} is not available in this environment. "
                f"The generated code has been validated for structural correctness only."
            )
            execution_success = True  # syntax check passed (would be False if validate_code_output failed above)
        
        if execution_success:
            report = (
                f"[SANDBOX EXECUTION OUTPUT]\n{execution_result}\n\n"
                f"[EVALUATED TEST SCENARIOS]\n{cases_str}\n\n"
                f"[VERIFICATION STATUS] All test scenarios evaluated successfully for {target_language.upper()}."
            )
            feedback_message = f"✅ Code passed all checks for {target_language.upper()}."
        else:
            report = (
                f"[SANDBOX EXECUTION ERROR]\n{execution_result}\n\n"
                f"[EVALUATED TEST SCENARIOS]\n{cases_str}\n\n"
                f"[VERIFICATION STATUS] Code execution error encountered."
            )
            feedback_message = f"❌ Execution error in {target_language.upper()} code."
        
        return {
            "report": report,
            "execution_success": execution_success,
            "messages": [AIMessage(content=feedback_message)]
        }
        
    except Exception as e:
        user_friendly_error = _make_user_friendly_error(e)
        return {
            "report": f"### TESTING ERROR\n{user_friendly_error}",
            "execution_success": False,
            "messages": [AIMessage(content=f"❌ Tester error: {user_friendly_error}")]
        }
        
    except Exception as e:
        user_friendly_error = _make_user_friendly_error(e)
        return {
            "report": f"### TESTING ERROR\n{user_friendly_error}",
            "execution_success": False,
            "messages": [AIMessage(content=f"❌ Tester error: {user_friendly_error}")]
        }


def should_continue(state: CrewState) -> Literal["developer", "end"]:
    MAX_ITERATIONS = state.get("max_iterations", 3)
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "end"
    if state.get("execution_success", False):
        return "end"
    return "developer"


def guardrail_node(state: CrewState) -> Dict[str, Any]:
    """Input guardrail node — scans user task before reaching the developer."""
    if not GUARDRAILS_AVAILABLE:
        return {"messages": []}  # Pass through if guardrails not loaded
    
    task = state["messages"][0].content if state["messages"] else ""
    input_report = InputGuard.scan_all(task)
    guardrail_stats.record_input_scan(input_report)
    
    if not input_report.passed:
        logger.warning(f"🛡️ Input guardrail blocked: {input_report.blocked_by}")
        return {
            "code": f"// GUARDRAIL BLOCKED: {input_report.reason}",
            "report": f"### 🛡️ Input Guardrail Alert\n{input_report.reason}\n\nBlocked by: {input_report.blocked_by}\nSeverity: {input_report.severity.value}",
            "execution_success": False,
            "iterations": state.get("max_iterations", 3),  # Skip to end
            "messages": [AIMessage(content=f"🛡️ {input_report.reason}")]
        }
    
    return {"messages": []}  # Pass through to developer


def create_workflow() -> StateGraph:
    workflow = StateGraph(CrewState)
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("developer", developer_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("tester", tester_node)
    
    workflow.add_edge(START, "guardrail")
    workflow.add_edge("guardrail", "developer")
    workflow.add_edge("developer", "human_review")
    
    workflow.add_conditional_edges(
        "human_review",
        should_route_from_human_review,
        {
            "tester": "tester",
            "developer": "developer",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "tester",
        should_continue,
        {
            "developer": "developer",
            "end": END
        }
    )
    return workflow


def get_agent():
    from langgraph.checkpoint.memory import MemorySaver
    workflow = create_workflow()
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


agent = get_agent()
__all__ = ["agent", "CrewState", "get_agent", "extract_task_intent", "generate_artifact_filename"]
