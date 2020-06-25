import json

import pandas as pd

from Models.Embeddings import Embedding2Array
from PreProcessing.Tools import PreProcess
from Processing.SentenceEmbeddings import SentenceEmbeddings
from Processing.SimilarityTools import SimilarityFunctions


def pre_processing(ques, ans):
    """
        Preprocess question and answer. Returns the filtered list of tokens
    :param ques: string
    :param ans: string
    :return: list
        Returns the filtered list after all preprocessing steps
    """
    preprocess = PreProcess()
    question_demoted = preprocess.question_demoting(ques, ans)
    filtered_sentence = preprocess.remove_stop_words(question_demoted)
    return filtered_sentence


if __name__ == '__main__':

    df = pd.read_csv('dataset/mohler_dataset_edited.csv')
    # columns = ['Unnamed: 0', 'id', 'question', 'desired_answer', 'student_answer',
    # 'score_me', 'score_other', 'score_avg']

    # Get the student answers from dataset
    student_answers = df['student_answer'].to_list()
    gpt2_similarity_score = {}

    # For each student answer, get id, question, desired answer
    for stu_ans in student_answers:
        id = df.loc[df['student_answer'] == stu_ans, 'id'].iloc[0]
        question = df.loc[df['student_answer'] == stu_ans, 'question'].iloc[0]
        desired_answer = df.loc[df['student_answer'] == stu_ans, 'desired_answer'].iloc[0]

        # Preprocess student answer
        pp_desired = pre_processing(question, desired_answer)
        pp_student = pre_processing(question, stu_ans)

        # Assign embeddings to desired answer and student answer
        embed2arr = Embedding2Array()

        word_array_1 = embed2arr.gpt2(pp_desired)
        word_array_2 = embed2arr.gpt2(pp_student)

        # Compare and assign cosine similarity to the answers

        similarity_tools = SimilarityFunctions(word_array_1, word_array_2)
        sentence_embed = SentenceEmbeddings()

        text_1_embed = sentence_embed.gpt_sowe(word_array_1)
        text_2_embed = sentence_embed.gpt_sowe(word_array_2)

        gpt2_similarity_score[stu_ans] = similarity_tools.get_cosine_similarity(text_1_embed, text_2_embed)
    print(gpt2_similarity_score)
    # Saving similarity scores to json
    with open('json_files/gpt2_similarity_score.json', 'w') as fp:
        json.dump(gpt2_similarity_score, fp)

    for answer in student_answers:
        df.loc[df['student_answer'] == answer, 'gpt2_sim_score'] = gpt2_similarity_score[answer]

    df.to_csv('dataset/mohler_dataset_edited.csv')