import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure()
ax = fig.add_subplot(111)

ANSWERS_PATH = 'dataset\mohler_dataset_edited.csv'
answers_data = pd.read_csv(ANSWERS_PATH)

count = answers_data['score_avg'].value_counts(sort=False)
hist = answers_data['score_avg'].hist(grid=False)

ax.set_xticks(np.arange(0, 5.5, 0.5))
ax.grid(linestyle='--')

plt.ylabel('count', fontsize=12, weight='bold')
plt.xlabel('assigned grade', fontsize=12, weight='bold')

plt.show()
