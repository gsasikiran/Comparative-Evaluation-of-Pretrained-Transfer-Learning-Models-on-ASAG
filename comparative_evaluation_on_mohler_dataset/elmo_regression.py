import numpy as np
import pandas as pd

from Processing.Stats import Metrics
from Training.Regression import RegressionAnalysis

df = pd.read_csv('dataset/answers_with_similarity_score.csv')


def train_test_split(data, percentage):
    msk = np.random.rand(len(data)) < (percentage / 100)
    data_train = df[msk]
    data_test = df[~msk]

    return data_train, data_test


def avg(given_list):
    return sum(given_list) / len(given_list)


def calculate_results(model_name):
    train_data, test_data = train_test_split(df, 70)
    model_name_score = 'normalized_' + model_name.lower() + '_score'
    train_data_x = train_data[model_name_score]
    train_data_y = train_data['score_avg']

    test_data_x = test_data[model_name_score]
    test_data_y = test_data['score_avg'].to_list()

    regression = RegressionAnalysis(train_data_x, train_data_y, test_data_x)

    test_y_pred_lin = [float(x) for x in regression.linear()]
    test_y_pred_rid = [float(x) for x in regression.ridge()]
    test_y_pred_iso = list(np.nan_to_num(regression.isotonic(), nan=0))

    metrics_iso = Metrics(test_data_y, test_y_pred_iso)
    metrics_lin = Metrics(test_data_y, test_y_pred_lin)
    metrics_rid = Metrics(test_data_y, test_y_pred_rid)

    return metrics_iso.rmse(), metrics_iso.pearson_correlation(), metrics_lin.rmse(), metrics_lin.pearson_correlation(), metrics_rid.rmse(), metrics_rid.pearson_correlation()


if __name__ == '__main__':

    name = str(input('Enter the model name(bert, elmo, gpt, gpt2) to calculate results: '))

    iso_rmse = []
    iso_pearson = []

    lin_rmse = []
    lin_pearson = []

    rid_rmse = []
    rid_pearson = []

    for i in range(0, 1000):
        iso_rmse_score, iso_pc_score, lin_rmse_score, lin_pc_score, rid_rmse_score, rid_pc_score = calculate_results(
            name)
        iso_rmse.append(iso_rmse_score)
        iso_pearson.append(iso_pc_score)

        lin_rmse.append(lin_rmse_score)
        lin_pearson.append(lin_pc_score)

        rid_rmse.append(rid_rmse_score)
        rid_pearson.append(rid_pc_score)

    print('Metric \t \t \t | Isotonic Regression \t | Linear Regression \t | Ridge Regression | ')
    print('------------------------------------------------------------------------------------------------')
    print('RMSE \t \t | ', round(avg(iso_rmse), 3), '\t |', round(avg(lin_rmse), 3), '\t |', round(avg(rid_rmse), 3),
          ' |')
    print('Pearson Correlation \t | ', round(avg(iso_pearson), 3), '\t |', round(avg(lin_pearson), 3), '\t |',
          round(avg(rid_pearson), 3), ' |')
# print('Spearman Correlation \t | ', metrics_iso.spearman_correlation(),'\t |', metrics_lin.spearman_correlation(), '\t |', metrics_rid.spearman_correlation(), ' |')
