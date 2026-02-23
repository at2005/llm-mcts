import json
import re
import time
from math import isfinite
from numbers import Real
from redis import Redis
from simpleeval import simple_eval

class Graders:
    def __init__(self):
        self.redis = Redis(host="localhost", port=6379, db=0)
        self.positive_reward = 1.0
        self.negative_reward = -1.0
        self._missing_reward = 0.0
        self.format_reward = 0.1
        self._redis_retry_attempts = 3
        self._redis_retry_delay_s = 0.05
        self._countdown_prompt_meta_cache = {}

    def parse_answer(self, answer: str) -> str:
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        last_assistant_idx = answer.rfind(assistant_marker)
        search_text = answer[last_assistant_idx:] if last_assistant_idx != -1 else answer

        matches = re.findall(r"<answer>(.*?)</answer>", search_text, re.DOTALL)
        return matches[-1].strip() if matches else None

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

    def _read_countdown_prompt_meta(self, prompt_id: int):
        cached_meta = self._countdown_prompt_meta_cache.get(prompt_id)
        if cached_meta is not None:
            return cached_meta

        raw_meta = self._read_redis_with_retry(f"countdown_prompt_meta:{prompt_id}")
        if raw_meta is not None:
            try:
                payload = json.loads(raw_meta)
                correct_answer = payload.get("correct_answer")
                input_numbers = payload.get("input_numbers")
                meta = (correct_answer, input_numbers)
                self._countdown_prompt_meta_cache[prompt_id] = meta
                return meta
            except Exception:
                self._countdown_prompt_meta_cache[prompt_id] = (None, None)
                return None, None

        correct_answer = self._read_redis_with_retry(f"correct_answer:{prompt_id}")
        input_numbers_raw = self._read_redis_with_retry(f"input_numbers:{prompt_id}")
        if correct_answer is None or input_numbers_raw is None:
            self._countdown_prompt_meta_cache[prompt_id] = (None, None)
            return None, None

        meta = (correct_answer, input_numbers_raw)
        self._countdown_prompt_meta_cache[prompt_id] = meta
        return meta

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

    def get_countdown_prompt_metadata(self, prompt_id: int) -> tuple[int, list[int]]:
        correct_answer_raw, input_numbers_raw = self._read_countdown_prompt_meta(prompt_id)
        try:
            if correct_answer_raw is None or input_numbers_raw is None:
                return self._missing_reward
            correct_answer = int(correct_answer_raw)
            if isinstance(input_numbers_raw, str):
                input_numbers = json.loads(input_numbers_raw)
            else:
                input_numbers = input_numbers_raw
        except Exception as exc:
            return None, None

        return correct_answer, input_numbers

    def countdown_reward_func(self, model_answer: str, correct_answer: int, input_numbers: list[int], dense_rewards: bool = False) -> float:
        reward = 0.0
        model_expr = self.parse_answer(model_answer)
        if model_expr is None:
            return self.negative_reward
        reward += self.format_reward

        if not self.validate_numbers(model_expr, input_numbers):
            return self.negative_reward + reward

        try:
            answer = simple_eval(model_expr)
        except Exception as exc:
            return self.negative_reward + reward

        if isinstance(answer, bool) or not isinstance(answer, Real):
            return self.negative_reward + reward
        if isinstance(answer, float) and not isfinite(answer):
            return self.negative_reward + reward

        distance = abs(float(answer) - correct_answer)

        if not dense_rewards:
            return float(distance == 0)

        correct_reward = self.positive_reward - self.format_reward - min(distance / correct_answer, 1.0) * 2.0
        return correct_reward + reward

    def countdown_grader(self, model_answer: str, prompt_id: int) -> float:
        correct_answer, input_numbers = self.get_countdown_prompt_metadata(prompt_id)
        if correct_answer is None or input_numbers is None:
            return self._missing_reward
        return self.countdown_reward_func(model_answer, correct_answer, input_numbers)