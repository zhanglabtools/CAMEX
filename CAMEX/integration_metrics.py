# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/10/28 20:45
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : integration_metircs.py
# @IDE     : PyCharm
import scanpy as sc
import scanpy.external as sce
import scib
import pandas as pd
import os


def evaluate(adata_raw, adata_int, method_name, batch_key='batch', label_key='cell_ontology_class', cluster_nmi=None,
             verbose=False,
             ari_=True,  # biological
             nmi_=True,  # biological
             silhouette_=True,  # asw batch: batch, asw label: biological
             pcr_=True,  # batch
             isolated_labels_f1_=True,  # biological
             isolated_labels_asw_=True,  # biological
             graph_conn_=True,  # batch
             kBET_=True,  # batch，win下爆内存

             ):
    """
    ref: https://github.com/theislab/scib-pipeline/blob/main/scripts/metrics/metrics.py
    需要将算法的embedding存放在adata.obsm['X_emb']
    """

    if method_name in ['scanorama', 'harmony', 'pyliger', 'scvi', 'scanvi', 'scalex',
                       'cell_train_class', 'cell_train_hidden', 'cell_pretrain_hidden']:
        type_ = 'embed'
        embed = 'X_emb'
    elif method_name in ['bbknn']:
        type_ = 'knn'
        embed = 'X_pca'
    elif method_name in ['seurat']:
        type_ = 'full'
        embed = 'X_pca'
    else:
        raise NotImplementedError
    result = scib.me.metrics(
        adata_raw,  # 未整合前的数据
        adata_int,  # 整合后的数据
        batch_key=batch_key,  # 批次的key
        label_key=label_key,  # 细胞类型的key
        type_=type_,  # 整合方法类型, either knn, full or embed
        embed=embed,  # 整合后embedding的key
        # cluster_key='leiden',  # 聚类标签的key
        cluster_nmi=cluster_nmi,  # 计算各个resolution下nmi保存的文件
        verbose=verbose,  # 是否显示

        # 是否计算下列评价指标
        ari_=ari_,  # biological
        nmi_=nmi_,  # biological
        silhouette_=silhouette_,  # asw batch: batch, asw label: biological
        pcr_=pcr_,  # batch
        isolated_labels_f1_=isolated_labels_f1_,  # biological
        isolated_labels_asw_=isolated_labels_asw_,  # biological
        graph_conn_=graph_conn_,  # batch
        kBET_=kBET_,  # batch，win下爆内存
        clisi_=False,  # biological，32位程序不能用
        ilisi_=False,  # batch，32位程序不能用

        trajectory_=False,
        cell_cycle_=False,
        hvg_score_=False,
    )
    return result  # df


if __name__ == '__main__':
    print('hello world')
