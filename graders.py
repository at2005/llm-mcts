import re
from redis import Redis

class Graders:
    def __init__(self):
        self.redis = Redis(host="localhost", port=6379, db=0)

    def maths_grader(self, string_state: str, prompt_id: int) -> float:
        print("Grading prompt: ", prompt_id)
        answer = re.search(r'<answer>(.*?)</answer>', string_state).group(1)
        print("Answer: ", answer)
        correct_answer = self.redis.get(f"correct_answer:{prompt_id}")
        print("Correct answer: ", correct_answer)
        if correct_answer is None:
            raise ValueError(f"Correct answer not found for prompt {prompt_id}")
        
        if answer == correct_answer:
            return 1.0
        else:
            return 0.0