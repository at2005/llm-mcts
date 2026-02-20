import json
import re
import time

from redis import Redis
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
from simpleeval import simple_eval


class Graders:
    def __init__(self):
        self.redis = Redis(host="localhost", port=6379, db=0)
        self.positive_reward = 1.0
        self.negative_reward = -1.0
        self._missing_reward = 0.0
        self.format_reward = 0.0
        self._redis_retry_attempts = 3
        self._redis_retry_delay_s = 0.05

    def parse_answer(self, answer: str) -> str:
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        last_assistant_idx = answer.rfind(assistant_marker)
        search_text = answer[last_assistant_idx:] if last_assistant_idx != -1 else answer

        matches = re.findall(r"<answer>(.*?)</answer>", search_text, re.DOTALL)
        return matches[-1].strip() if matches else None

    def _normalize_answer_for_parser(self, answer: str) -> str:
        normalized = answer.strip()
        if normalized.count("$") == 1:
            normalized = normalized.replace("$", "")
        return normalized

    def _read_redis(self, key: str) -> str:
        try:
            value = self.redis.get(key)
        except Exception as exc:
            raise RuntimeError(f"Failed to read {key} from Redis") from exc
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return value.strip()

    def _read_redis_with_retry(self, key: str) -> str:
        for attempt in range(self._redis_retry_attempts + 1):
            try:
                value = self.redis.get(key)
            except Exception:
                value = None

            if value is not None:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                return value.strip()

            if attempt < self._redis_retry_attempts:
                time.sleep(self._redis_retry_delay_s)
        return None

    def maths_grader(self, string_state: str, prompt_id: int) -> float:
        key = f"correct_answer:{prompt_id}"
        gold_answer = self._read_redis_with_retry(key)

        if gold_answer is None:
            return self._missing_reward

        model_answer = self.parse_answer(string_state)
        model_answer = self._normalize_answer_for_parser(model_answer)
        try:
            extraction = [LatexExtractionConfig()]
            gold = parse(gold_answer, extraction_config=extraction, parsing_timeout=None)
            pred = parse(model_answer, extraction_config=extraction, parsing_timeout=None)
            return (
                self.positive_reward
                if verify(gold, pred, timeout_seconds=None)
                else self.negative_reward
            )
        except Exception:
            return self.negative_reward

    def gsm8k_grader(self, string_state: str, prompt_id: int) -> float:
        reward = 0.0
        answer = self.parse_answer(string_state)
        if answer is None:
            return self.negative_reward
        reward += self.format_reward

        correct_answer = self._read_redis(f"correct_answer:{prompt_id}")
        if correct_answer is None:
            raise ValueError(f"Correct answer not found for prompt {prompt_id}")

        correct_answer = str(correct_answer).strip()

        return (
            self.positive_reward + reward
            if answer == correct_answer
            else self.negative_reward + reward
        )

    def validate_numbers(self, expression: str, input_numbers: list[int]) -> bool:
        used_numbers = [int(n) for n in re.findall(r'\d+', expression)]
        available = list(input_numbers)
        for n in used_numbers:
            if n in available:
                available.remove(n)
            else:
                return False
        return True

    def countdown_grader(self, model_answer: str, prompt_id: int) -> float:
        reward = 0.0
        model_expr = self.parse_answer(model_answer)
        if model_expr is None:
            return self.negative_reward
        reward += self.format_reward

        correct_answer = self._read_redis(f"correct_answer:{prompt_id}")
        input_numbers_raw = self._read_redis(f"input_numbers:{prompt_id}")

        try:
            correct_answer = int(correct_answer)
            input_numbers = json.loads(input_numbers_raw)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to process answer data for prompt {prompt_id}"
            ) from exc

        if not self.validate_numbers(model_expr, input_numbers):
            return self.negative_reward + reward

        answer = simple_eval(model_expr)

        if answer != correct_answer:
            return self.negative_reward + reward

        return self.positive_reward + reward
