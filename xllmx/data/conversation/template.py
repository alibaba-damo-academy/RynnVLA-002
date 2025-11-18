from typing import List


class ConversationBase:
    roles = ["Human", "Assistant"]

    def __init__(self, messages=None):
        self.messages = messages or []

    def process(self):
        raise NotImplementedError

    def get_prompt(self):
        return self.process()["conv"]

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return ConversationBase(
            messages=[[x, y] for x, y in self.messages],
        )

    def load_qas(self, qas: List[List[str]]):
        self.messages = []
        for q, a in qas:
            self.append_message(self.roles[0], q)
            self.append_message(self.roles[1], a)
