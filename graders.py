import re
from redis import Redis
from math_verify import parse, verify, LatexExtractionConfig

class Graders:
    def __init__(self):
        self.redis = Redis(host="localhost", port=6379, db=0)
        self.positive_reward = 1.0
        self.negative_reward = -1.0
    
    def parse_answer(self, answer: str) -> str:
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        last_assistant_idx = string_state.rfind(assistant_marker)
        if last_assistant_idx != -1:
            search_text = string_state[last_assistant_idx:]
        else:
            search_text = string_state

        matches = re.findall(r"<answer>(.*?)</answer>", search_text, re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()

    def maths_grader(self, string_state: str, prompt_id: int) -> float:
        gold_answer = self.redis.get(f"correct_answer:{prompt_id}")
        model_answer = self.parse_answer(string_state)
        try:
            gold = parse(gold_answer, extraction_config=[LatexExtractionConfig()])
            pred = parse(model_answer, extraction_config=[LatexExtractionConfig()])
            return self.positive_reward if verify(gold, pred) else self.negative_reward
        except Exception:
            return self.negative_reward


    def gsm8k_grader(self, string_state: str, prompt_id: int) -> float:
        reward = 0.0
        answer = self.parse_answer(string_state)
        if answer is None:
            return self.negative_reward
        else:
            reward += 0.1

        try:
            correct_answer = self.redis.get(f"correct_answer:{prompt_id}")
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
            return self.positive_reward + reward
        else:
            return self.negative_reward + reward
