import re
from redis import Redis

class Graders:
    def __init__(self):
        self.redis = Redis(host="localhost", port=6379, db=0)

    def maths_grader(self, string_state: str, prompt_id: int) -> float:
        answer = re.search(r'<answer>(.*?)</answer>', string_state).group(1)
        correct_answer = self.redis.get(f"correct_answer:{prompt_id}")
        if correct_answer is None:
            raise ValueError(f"Correct answer not found for prompt {prompt_id}")
        
        if answer == correct_answer:
            return 1.0
        else:
            return 0.0