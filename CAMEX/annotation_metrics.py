# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/8/13 10:35
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : script_intagration.py
# @IDE     : PyCharm

"""
evaluation on multi-class metrics
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from CAMEX.postprocess_untils import plot_sankey
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# matplotlib.use('TkAgg')


def get_cm(y_true, y_pred):
    """
    ref:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    https://blog.csdn.net/SartinL/article/details/105844832
    行为真实标签，列为预测标签
    :param y_true:
    :param y_pred:
    :return: cm
    """
    # calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=None)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    # disp.plot()
    # plt.show()
    return cm


def get_balanced_accuracy(y_true, y_pred, r=4):
    """
    ref:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    :param y_true:
    :param y_pred:
    :param r:
    :return:
    """
    score_acc = round(balanced_accuracy_score(y_true, y_pred), r)
    return score_acc


def get_precision(y_true, y_pred, average='micro', r=4):
    """
    ref:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

    :param y_true:
    :param y_pred:
    :param average:
    :param r:
    :return:
    """
    assert average in {'micro', 'macro', 'weighted'}
    score_acc = round(precision_score(y_true, y_pred, average=average), r)
    return score_acc


def get_recall(y_true, y_pred, average='micro', r=4):
    """
    ref:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

    :param y_true:
    :param y_pred:
    :param average:
    :param r:
    :return:
    """
    assert average in {'micro', 'macro', 'weighted'}
    score = round(precision_score(y_true, y_pred, average=average), r)
    return score


def get_f1(y_true, y_pred, average='micro', r=4):
    """
    ref:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    :param y_true:
    :param y_pred:
    :param average:
    :param r:
    :return:
    """
    assert average in {'micro', 'macro', 'weighted'}
    score = round(f1_score(y_true, y_pred, average=average), r)
    return score


def get_auroc(y_true, y_pred_prob, average='macro', r=4):
    """
    ref:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    example:
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(solver="liblinear").fit(X, y)
    roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')

    :param y_true:
    :param y_pred_prob:
    :param average:
    :param r:
    :return:
    """
    assert average in {'macro', 'weighted'}
    score = round(roc_auc_score(y_true, y_pred_prob, multi_class='ovr'), r)
    return score


def get_aupr(y_true, y_pred_prob, r=4, total_label=None):
    """
    ref:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    :param y_true:
    :param y_pred_prob:
    :param r:
    :param total_label:
    :return:
    """
    # calculate total type of label
    if total_label is None:
        total_label = len(np.unique(y_true))
    # binary
    y_true = label_binarize(y_true, classes=np.unique(y_true))
    # toarray
    y_pred_prob = np.array(y_pred_prob)
    # for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(total_label):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred_prob[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true.ravel(), y_pred_prob.ravel()
    )
    average_precision["micro"] = average_precision_score(y_true, y_pred_prob, average="micro")
    # # plot NotImplemented
    # pl = True
    # if pl:
    #
    #     display = PrecisionRecallDisplay(
    #         recall=recall["micro"],
    #         precision=precision["micro"],
    #         average_precision=average_precision["micro"],
    #     )
    #     display.plot()
    #     _ = display.ax_.set_title("Micro-averaged over all classes")
    #     plt.show()
    return round(average_precision["micro"], r)


def evaluate_metrics(y_true, y_pred, y_pred_prob, average='weighted', r=6):
    result_temp = {}
    result_temp['acc'] = get_balanced_accuracy(y_true, y_pred, r=r)
    result_temp['precision'] = get_precision(y_true, y_pred, average=average, r=r)
    result_temp['recall'] = get_recall(y_true, y_pred, average=average, r=r)
    result_temp['f1'] = get_f1(y_true, y_pred, average=average, r=r)
    result_temp['auroc'] = get_auroc(y_true, y_pred_prob, average=average, r=r)
    result_temp['aupr'] = get_aupr(y_true, y_pred_prob, r=r)
    # result_temp = pd.DataFrame.from_dict(result_temp, orient="index")
    # result_temp.columns = [method_name]
    return result_temp


def evaluate_all(y_true, y_prob, y_pred, y_ontology, cell_ontology_dict, cell_class_dict, method, dataset_name):

    # mkdir
    result_file = 'annotation_results'
    if os.path.exists(result_file):
        pass
    else:
        os.mkdir(result_file)

    cell_ontology_list = [cell_ontology_dict[i] for i in range(len(cell_ontology_dict))]
    cell_class_list = [cell_class_dict[i] for i in range(len(cell_class_dict))]
    # 计算混淆矩阵cm
    # 1 标准化的cm
    cm_unify = get_cm(np.concatenate([y_true, np.arange(len(cell_class_dict))]),
                      np.concatenate([y_pred, np.arange(len(cell_class_dict))]))  # y_true为标准化到train的标签
    # 主对角线减1
    for i, item in enumerate(cm_unify):
        item[i] -= 1
    # df
    cm_unify_df = pd.DataFrame(cm_unify, index=cell_class_list, columns=cell_class_list)
    # plot heatmap
    ax = sns.heatmap(cm_unify_df, annot=True, fmt="d", linewidths=0.2, cmap="YlGnBu", cbar=False)
    ax.figure.set_size_inches(8, 8)
    plt.xticks(fontsize=6, rotation=30)
    plt.yticks(fontsize=6, rotation=30)
    plt.savefig(f'./annotation_results/{method}_{dataset_name}_cm_unify.jpg', dpi=600)
    # plt.show()
    del ax

    # 2 未标准化的cm，通过y_ontology和y_pred计算，使用对角阵填充
    cm_raw = get_cm(
        np.concatenate([y_ontology, np.arange(len(cell_ontology_dict))]),
        np.concatenate([y_pred, np.arange(len(cell_ontology_dict))]))  # y_true为真实值
    # 主对角线减1
    for i, item in enumerate(cm_raw):
        item[i] -= 1
    cm_raw_df = pd.DataFrame(cm_raw, index=cell_ontology_list, columns=cell_ontology_list)
    # 行是source，列是target，去掉和为0的行（没有这些细胞类型），去掉unknown以后的列（不会被预测到）
    # 去掉行
    cm_raw_df = cm_raw_df.loc[cm_raw_df.sum(axis=0) != 0, :]
    # 去掉列
    cm_raw_df = cm_raw_df.loc[:, :'unknown']
    # plot
    # ax = sns.heatmap(cm_raw_df, fmt="d", linewidths=0.2, cmap="YlGnBu_r", cbar=False, annot=True)
    ax = sns.heatmap(cm_raw_df, fmt="d", linewidths=0.2, cmap="YlGnBu", cbar=False, annot=True)
    ax.figure.set_size_inches(8, 8)
    plt.xticks(fontsize=6, rotation=30)
    plt.yticks(fontsize=6, rotation=30)
    plt.savefig(f'./annotation_results/{method}_{dataset_name}_cm_raw.jpg', dpi=600)
    # plt.show()
    del ax

    # # plot sanky 绘制桑基图，由混淆矩阵绘制桑基图
    # # unify
    # fig = plot_sankey(cm_unify_df)
    # # fig.show()
    # fig.write_image(f'./annotation_results/{method}_{dataset_name}_sanky_unify.png', width=600, height=800, scale=5)
    # # raw
    # fig = plot_sankey(cm_raw_df)    # 全部的忽略
    # # fig.show()
    # fig.write_image(f'./annotation_results/{method}_{dataset_name}_sanky_raw.png', width=600, height=800, scale=5)
    # # metrics
    # # test中有些未出现在train，无法计算auroc，aupr，为y_true，y_pred，y_pred_prob加一个对角阵
    # y_true = np.concatenate([y_true, np.arange(0, len(cell_class_dict))])
    # y_prob = np.concatenate([y_prob, np.eye(len(cell_class_dict))])
    # y_pred = np.argmax(y_prob, axis=-1)

    # evaluate_metrics
    result_metrics = evaluate_metrics(y_true, y_pred, y_prob)

    return cm_unify, cm_raw, result_metrics


if __name__ == '__main__':
    pass
    # """
    # 由于unknown的存在，某些指标无法计算，所以把unknown那一列去掉
    # """
    # x = np.array([[1, 3], [2, 6]])
    # x = x / x.sum(axis=1, keepdims=1)
    # print()
    #
    # # example
    # # 混淆矩阵，行是y_true，列是y_pred
    # y_true = [0, 1, 2, 1]
    # y_pred = [0, 1, 2, 2]
    # y_prob = [[1, 0, 0],
    #           [0, 1, 0],
    #           [0, 0, 1],
    #           [0, 0, 1]]
    # print(evaluate_all('came', y_true, y_pred, y_prob))
    # print()
