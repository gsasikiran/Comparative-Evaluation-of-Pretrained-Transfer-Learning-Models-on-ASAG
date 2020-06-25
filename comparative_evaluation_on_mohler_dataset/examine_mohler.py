import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANSWERS_PATH = 'dataset\Mohler_dataset_tim\cleaned\mohler_answers.csv'
answers_data = pd.read_csv(ANSWERS_PATH)

score_avg = answers_data['score_avg'].to_list()
scores_allotted = answers_data['score_avg'].unique()
scores_allotted = np.sort(scores_allotted)

count = []
for score in scores_allotted:
    count.append(score_avg.count(score))

plt.plot(scores_allotted, count)
plt.show()
