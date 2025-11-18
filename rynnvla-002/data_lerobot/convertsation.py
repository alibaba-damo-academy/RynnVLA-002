from typing import List


class Conversation:
    sep_token = "<reserved08706>"
    roles = ["Human", "Assistant"]

    def __init__(self, messages=None):
        self.messages = messages or []

    def process(self):
        ret = ""
        pieces = []
        for i, (role, message) in enumerate(self.messages):
            if message is not None:
                turn = message + self.sep_token
                ret += turn
                if role == self.roles[1]:
                    pieces.append({"data": turn, "predict": True})
                else:
                    pieces.append({"data": turn, "predict": False})
            else:
                # generation prompt
                assert i == len(self.messages) - 1 and role == self.roles[1], "only last assistant message can be None"

        result = {
            "conv": ret,  # text involving the complete conversation
            "pieces": pieces,  # list to help correctly mark the labels
        }
        return result

    def get_prompt(self):
        return self.process()["conv"]

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            messages=[[x, y] for x, y in self.messages],
        )

    def load_qas(self, qas: List[List[str]]):
        """
        convert the list of question-answer pairs to a string, which contains the conversation involving all
          the questions and answers. When the last answer is None, the returned string is the prompt which
          can be used by the model to generate the last answer.
        :param qas: [[question1, answer1], [question2, answer2], ..., [questionX, answerX]]
          note that the last answer, i.e. answerX, can be None
        :return: the prompt
        """
        self.messages = []
        for q, a in qas:
            self.append_message(self.roles[0], q)
            self.append_message(self.roles[1], a)
