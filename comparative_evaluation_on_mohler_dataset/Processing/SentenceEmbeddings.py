import scipy


class SentenceEmbeddings:
    def __init__(self):
        pass

    def sowe(self, array):

        return sum(array)

    def mowe(self, array):

        return sum(array)/len(array)

    def gpt_sowe(self, array):

        return [sum(i) for i in zip(*array)]
