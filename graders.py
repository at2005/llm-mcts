import re
import time
from redis import Redis
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

class Graders:
    def __init__(self):
        self.redis = Redis(host="localhost", port=6379, db=0)
        self.positive_reward = 1.0
        self.negative_reward = -1.0
        self._missing_reward = 0.0
        self._redis_retry_attempts = 3
        self._redis_retry_delay_s = 0.05
    
    def parse_answer(self, answer: str) -> str:
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        last_assistant_idx = answer.rfind(assistant_marker)
        if last_assistant_idx != -1:
            search_text = answer[last_assistant_idx:]
        else:
            search_text = answer 

        matches = re.findall(r"<answer>(.*?)</answer>", search_text, re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()

    def _normalize_answer_for_parser(self, answer: str) -> str:
        if answer is None:
            return None
        normalized = answer.strip()
        if normalized.count("$") == 1:
            normalized = normalized.replace("$", "")
        return normalized

    def maths_grader(self, string_state: str, prompt_id: int) -> float:
        key = f"correct_answer:{prompt_id}"

        gold_answer = None
        for attempt in range(self._redis_retry_attempts + 1):
            try:
                # print(f"Reading correct answer for prompt {prompt_id} from Redis")
                gold_answer = self.redis.get(key)
                # print(f"Correct answer for prompt {prompt_id} from Redis: {gold_answer}")
            except Exception as e:
                # print(f"Error reading correct answer for prompt {prompt_id} from Redis: {e}")
                gold_answer = None

            if gold_answer is not None:
                break

            if attempt < self._redis_retry_attempts:
                time.sleep(self._redis_retry_delay_s)

        if gold_answer is None:
            # print(f"No correct answer found for prompt {prompt_id}")
            return self._missing_reward
        if isinstance(gold_answer, bytes):
            gold_answer = gold_answer.decode("utf-8", errors="ignore")
        else:
            gold_answer = str(gold_answer)

        model_answer = self.parse_answer(string_state)
        model_answer = self._normalize_answer_for_parser(model_answer)
        # print(f"Model answer: {model_answer}")
        try:
            extraction = [LatexExtractionConfig()]
            gold = parse(gold_answer, extraction_config=extraction, parsing_timeout=None)
            pred = parse(model_answer, extraction_config=extraction, parsing_timeout=None)
            # print(f"Gold answer: {gold}")
            # print(f"Pred answer: {pred}")
            return (
                self.positive_reward
                if verify(gold, pred, timeout_seconds=None)
                else self.negative_reward
            )
        except Exception as e:
            # print(f"Error parsing answers: {e}")
            return self.negative_reward


    def gsm8k_grader(self, string_state: str, prompt_id: int) -> float:
        # print("Running GSM8K grader")
        reward = 0.0
        answer = self.parse_answer(string_state)
        if answer is None:
            return self.negative_reward
        else:
            reward += 0.1

        try:
            correct_answer = self.redis.get(f"correct_answer:{prompt_id}")
            # print(f"Correct answer for prompt {prompt_id} from Redis: {correct_answer}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read correct answer for prompt {prompt_id} from Redis"
            ) from exc
        
        if correct_answer is None:
            raise ValueError(f"Correct answer not found for prompt {prompt_id}")

        if isinstance(correct_answer, bytes):
            try:
                correct_answer_str = correct_answer.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Correct answer for prompt {prompt_id} is not valid UTF-8"
                ) from exc
        else:
            correct_answer_str = str(correct_answer).strip()

        if answer == correct_answer_str:
            # print(f"Answer is correct for prompt {prompt_id}")
            return self.positive_reward + reward
        else:
            # print(f"Answer is incorrect for prompt {prompt_id}")
            return self.negative_reward + reward
