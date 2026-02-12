import re
from redis import Redis


class Graders:
    def __init__(self):
        self.redis = Redis(host="localhost", port=6379, db=0)

    def maths_grader(self, string_state: str, prompt_id: int) -> float:
        # The decoded state contains the whole prompt + model output.
        # Extract only the last assistant turn to avoid matching <answer> tags
        # in system instructions or earlier turns.
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        last_assistant_idx = string_state.rfind(assistant_marker)
        if last_assistant_idx != -1:
            search_text = string_state[last_assistant_idx:]
        else:
            search_text = string_state

        matches = re.findall(r"<answer>(.*?)</answer>", search_text, re.DOTALL)
        if not matches:
            return 0.0
            #raise ValueError("No <answer>...</answer> block found in grader input")

        answer = matches[-1].strip()
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

        print("Grading prompt: ", prompt_id)
        print("Answer: ", answer)
        print("Correct answer: ", correct_answer_str)
        if answer == correct_answer_str:
            return 1.0
        else:
            return 0.0
