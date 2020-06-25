from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class PreProcess:
    def __init__(self):
        pass

    def tokenization(self, question, answer):
        question_tokens = word_tokenize(question)
        answer_tokens = word_tokenize(answer)
        return question_tokens, answer_tokens

    def question_demoting(self, question, answer):

        question_tokens, answer_tokens = self.tokenization(question, answer)
        demoted_tokens = [word for word in answer_tokens if word not in question_tokens]
        return demoted_tokens

    def remove_stop_words(self, demoted_tokens):

        stop_words = set(stopwords.words('english'))
        filtered_sentence = [w for w in demoted_tokens if not w in stop_words]
        return filtered_sentence


