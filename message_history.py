import dataclasses
from typing import List, Dict


@dataclasses.dataclass
class GPTMessage:
    role: str
    content: str


class History:
    def __init__(self, num_output=10):
        self.messages = []
        self.num_output = num_output

    def add(self, message: GPTMessage):
        self.messages.append(message)

    def get_messages(self) -> List[Dict[str, str]]:
        """Get the last N messages in OpenAI format"""
        return [
            dataclasses.asdict(message) for message in self.messages[-self.num_output:]
        ]