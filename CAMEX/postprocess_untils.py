# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/9/4 20:08
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : postprocess_script.py
# @IDE     : PyCharm

"""
对结果的处理，主要包含绘图
"""

import pandas as pd
import numpy as np
import copy
from IPython.display import HTML

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def get_color(num: int, transparency=0.8):
    """
    ref: https://plotly.com/python/sankey-diagram/
    :param num:
    :param transparency:
    :return:
    """
    if num <= 0:
        raise NotImplementedError
    else:
        # 准备10种颜色
        color_0 = f"rgba(245,148,148, {transparency})"
        color_1 = f'rgba(241,158,102, {transparency})'
        color_2 = f'rgba(250,208,98, {transparency})'
        color_3 = f'rgba(162,210,120, {transparency})'
        color_4 = f'rgba(206,182,215, {transparency})'
        color_5 = f'rgba(144,210,172, {transparency})'
        color_6 = f'rgba(131,177,215, {transparency})'
        color_7 = f'rgba(181,155,155, {transparency})'
        color_8 = f'rgba(215,214,214, {transparency})'
        color_9 = f'rgba(214,21,25, {transparency})'
        color_10 = f"rgba(241,153,17, {transparency})"
        color_11 = f'rgba(253,241,30, {transparency})'
        color_12 = f'rgba(112,179,29, {transparency})'
        color_13 = f'rgba(3,149,102, {transparency})'
        color_14 = f'rgba(1,150,233, {transparency})'
        color_15 = f'rgba(66,1,109, {transparency})'
        color_16 = f'rgba(209,2,109, {transparency})'
        color_17 = f'rgba(254,123,132, {transparency})'
        color_18 = f'rgba(158,205,255, {transparency})'
        color_list = [color_0, color_1, color_2, color_3, color_4, color_5, color_6, color_7, color_8, color_9,
                      color_10, color_11, color_12, color_13, color_14, color_15, color_16, color_17, color_18]

        if num <= 19:
            return color_list[: num]

        # 前10种颜色固定，超过10种随机产生
        elif num > 19:
            for i in range(19, num):
                while True:
                    color_temp = f"rgba({np.random.randint(0, 255, 1)[0]}, {np.random.randint(0, 255, 1)[0]}, " \
                                 f"{np.random.randint(0, 255, 1)[0]}, {transparency})"
                    # 重复则继续随机产生
                    if color_temp not in color_list:
                        break
                color_list.append(color_temp)
            return color_list


def plot_sankey(cm_df, transparency=0.8):
    """
    输入一个df的混淆矩阵，行为真实值，列为预测值
    桑基图为从source（真实值）到target（预测值）
    :param transparency:
    :param cm_df:
    :return:
    """
    # label
    label_source = cm_df.index.tolist()
    label_source_dict = {item: i for i, item in enumerate(label_source)}
    label_target = cm_df.columns.tolist()
    label_target_dict = {item: i + len(label_source_dict) for i, item in enumerate(label_target)}
    label = label_source + label_target  # source + target

    # source and target
    # color
    color_list = get_color(max(len(label_source), len(label_target)), transparency)

    source = []
    target = []
    value = []
    color_link = []
    for i, source_temp in enumerate(cm_df.index):
        for j, target_temp in enumerate(cm_df.columns):
            source.append(label_source_dict[source_temp])
            target.append(label_target_dict[target_temp])
            value.append(cm_df.loc[source_temp, target_temp])
            color_link.append(color_list[i])

    color_node = color_list[: len(label_source)] + color_list[: len(label_target)]

    # paint args
    # source到target的数量，与source和target一一对应，越大越粗
    link = dict(source=source, target=target, value=value, color=color_link)
    # label的修饰，pad：每个label上下间距，thickness：左右间距
    node = dict(label=label, pad=15, thickness=10, color=color_node, line=dict(color="black", width=0.5))

    # paint
    data = go.Sankey(link=link, node=node, arrangement="snap")  # plot
    fig = go.Figure(data)
    fig.update_layout(width=500)  # 调整图片宽度，参考 https://github.com/plotly/plotly.py/issues/2324
    # fig.write_image("./confluency_graph.png")     # 如何保存，参考 https://github.com/plotly/plotly.py/issues/3744
    return fig



def plot_confusion_matrix(cm, cell_type):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cell_type)
    disp.plot()
    plt.show()
    return


def plot_radar():
    pass


def plot_pie(labels, values):
    """
    ref: Styled Pie Chart in https://plotly.com/python/pie-charts/
    input labels and values, output pie figure
    :param labels: ['Oxygen', 'Hydrogen', 'Nitrogen', 'Oxygen1', 'Hydrogen1', 'Nitrogen1']
    :param values: values = [4500, 2500, 1053, 500, 4500, 2500]
    :return:
    """
    color_list = get_color(len(labels))
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    # 一些参数
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=color_list, line=dict(color='#000000', width=2)))
    # 画面宽度
    fig.update_layout(width=800)
    fig.show()


def plot_bar_sidle(y_data, x_data, top_y):
    """
    ref https://plotly.com/python/horizontal-bar-charts/
    :param y_data: 横坐标标签,     y_data = ['huamn', 'mouse', 'monkey']
    :param x_data: 横坐标标签所占的比例
        x_data = [[21, 30, 21, 16, 12],
              [24, 31, 19, 15, 11],
              [27, 26, 23, 11, 13],
              [29, 24, 15, 18, 14]]
    :param top_y: 每个比例对应的段标签    top_labels = ['a cell', 'b cell', 'c cell', 'd cell', 'long<br>cell']
    :return:
    """
    # 取颜色
    color_list = get_color(len(top_y))
    # 归一化并取证
    x_data = [np.around(item / item.sum() * 100, 2) for item in x_data]
    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=color_list[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=120, r=10, t=140, b=80),
        showlegend=False,
    )

    annotations = []

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(dict(xref='paper', yref='y',
                                x=0.14, y=yd,
                                xanchor='right',
                                text=str(yd),
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        # labeling the first percentage of each bar (x_axis)
        annotations.append(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text=str(xd[0]) + '%',
                                font=dict(family='Arial', size=14,
                                          color='rgb(248, 248, 255)'),
                                showarrow=False))
        # labeling the first Likert scale (on the top)
        if yd == y_data[-1]:
            annotations.append(dict(xref='x', yref='paper',
                                    x=xd[0] / 2, y=1.1,
                                    text=top_y[0],
                                    font=dict(family='Arial', size=14,
                                              color='rgb(67, 67, 67)'),
                                    showarrow=False))
        space = xd[0]
        for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i] / 2), y=yd,
                                    text=str(xd[i]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i] / 2), y=1.1,
                                        text=top_y[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]

    fig.update_layout(annotations=annotations)

    fig.show()


def plot_box(x_data):
    """
    ref: https://plotly.com/python/box-plots/
    :param x_data:
        x_data = [[21, 30, 21, 16, 12],
              [24, 31, 19, 15, 11],
              [27, 26, 23, 11, 13],
              [29, 24, 15, 18, 14]]
    :return:
    """
    color_list = get_color(len(x_data))
    fig = go.Figure([go.Box(y=item, marker_color=color_list[i]) for i, item in enumerate(x_data)])
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(zeroline=False, gridcolor='white'),
        paper_bgcolor='rgb(233,233,233)',
        plot_bgcolor='rgb(233,233,233)',
    )
    fig.show()


def plot_violin(methods, data):
    """
    ref https://plotly.com/python/violin/
    :param method: methods = ['method1', 'method2', 'method3', 'method4']
    :param data:
        x_data = [[21, 30, 21, 16, 12],
              [24, 31, 19, 15, 11],
              [27, 26, 23, 11, 13],
              [29, 24, 15, 18, 14]]
    :return:
    """
    fig = go.Figure()
    for i, item in enumerate(data):
        fig.add_trace(go.Violin(x=[methods[i]] * len(item),
                                y=data,
                                name=methods[i],
                                box_visible=True,
                                meanline_visible=True))
    fig.show()


def plot_bar_horizontal(coordinate, methods, performance):
    """
    ref https://plotly.com/python/horizontal-bar-charts/
    :param coordinate: ['brain', 'lung', 'tissue']
    :param methods: ['came', 'seurat']
    :param performance: [[99, 98, 99], [90, 89, 90]]
    :return:
    """
    fig = go.Figure()
    color_list = get_color(len(methods))
    for i, method in enumerate(methods):
        fig.add_trace(go.Bar(
            x=coordinate,
            y=performance[i],
            name=method,
            marker_color=color_list[i]
        ))
    fig.update_layout(barmode='group', xaxis_tickangle=-45)
    fig.show()


if __name__ == '__main__':
    print('hello world')
    print('some test scripts')
