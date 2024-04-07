# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/10/27 23:35
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : integration_method.py
# @IDE     : PyCharm
import scanpy as sc
import scanpy.external as sce
import scib
import pandas as pd
import os
import scib
import os
import numpy as np


def pyliger_integrate(adata):
    """
    https://github.com/welch-lab/pyliger/blob/master/integrating_multi_scRNA_data.ipynb
    pyliger需要做基因选择所以输入的是raw
    :param adata:
    :return:
    """

    import pyliger

    # 输入concat后的adata
    adata = adata.copy()
    # pyliger requirement
    adata.obs.index.name = 'barcodes'
    # adata.obs.index.name = 'index'
    adata.var.index.name = 'genes'
    # adata.var.index.name = 'index'
    batch_list = adata.obs.loc[:, 'batch'].unique()
    adata_list = [adata[adata.obs.loc[:, 'batch'] == item, :].copy() for item in batch_list]
    for i, data in enumerate(adata_list):
        data.uns['sample_name'] = data.uns['data_order'][i]
    # pyliger
    ifnb_liger = pyliger.create_liger(adata_list, remove_missing=False)
    pyliger.normalize(ifnb_liger, remove_missing=False)
    pyliger.select_genes(ifnb_liger)
    pyliger.scale_not_center(ifnb_liger)
    pyliger.optimize_ALS(ifnb_liger, k=30)
    pyliger.quantile_norm(ifnb_liger)
    # adata
    h_norm = np.vstack([adata.obsm['H_norm'] for adata in ifnb_liger.adata_list])
    adata.obsm['X_emb'] = h_norm
    return adata


def scvi_integrate(adata_int):
    from scvi.model import SCVI
    SCVI.setup_anndata(adata_int, layer="counts", batch_key='batch')
    # SCVI.setup_anndata(adata_int, batch_key='batch')
    vae = SCVI(
        adata_int,
        gene_likelihood="nb",
        n_layers=1,
        n_latent=10,
    )
    # vae.train()
    vae.train(train_size=1.0, max_epochs=100, use_gpu=True)
    adata_int.obsm["X_emb"] = vae.get_latent_representation()
    return adata_int, vae


# def seurat_integrate(adata):
#     from rpy2.robjects import r
#
#     # mkdir
#     try:
#         os.mkdir('./seurat')
#     except:
#         print('seurat file has been created')
#         pass
#
#     # save h5ad
#     batch = adata.obs.loc[:, 'batch'].unique()
#     batch_df = pd.DataFrame(batch)
#     batch_df.iloc[:, 0] = batch_df.iloc[:, 0].apply(lambda x: x + '.h5ad')
#     batch_df.to_csv('./adata_file.csv', index=False, header=False)
#     for item in batch:
#         adata_temp = adata[adata.obs.loc[:, 'batch'] == item, :]
#         adata_temp.write_h5ad('./' + item + '.h5ad')
#
#     # run seurat
#     rscript = """
#     print('start R')
#     rm(list=ls())
#     library(Seurat)
#     library(SeuratData)
#     library(zellkonverter)
#
#     # 读取
#     print('load data')
#     file_name <- read.table('adata_file.csv')[, 1]
#     adata_list <- list()
#     for (i in 1:length(file_name)) {
#       print(file_name[[i]])
#       data_temp <- readH5AD(file_name[[i]])
#       adata_temp <- as.Seurat(data_temp, counts = "X", data = NULL)
#       adata_list[i] <- adata_temp
#     }
#
#     # 预处理
#     print('preprocess')
#     for (i in 1:length(adata_list)) {
#       adata_list[[i]] <- NormalizeData(adata_list[[i]], verbose = FALSE)
#       adata_list[[i]] <- FindVariableFeatures(adata_list[[i]], selection.method = "vst", nfeatures = 2000,
#                                               verbose = FALSE)
#     }
#
#     # 整合2，大规模数据时用pca
#     print('integrate')
#     features <- SelectIntegrationFeatures(object.list = adata_list)
#     for (i in 1:length(adata_list)) {
#       adata_list[[i]] <- ScaleData(adata_list[[i]], features = features, verbose = FALSE)
#       adata_list[[i]] <- RunPCA(adata_list[[i]], features = features, verbose = FALSE)
#     }
#     anchors <- FindIntegrationAnchors(object.list = adata_list, reduction = "rpca", dims = 1:50)
#     integrated <- IntegrateData(anchorset = anchors, dims = 1:50)
#
#     # 保存
#     print('save')
#     library(SeuratDisk)
#     DefaultAssay(integrated) <- "integrated"
#     SaveH5Seurat(integrated, filename = "seurat_integrated.h5Seurat",overwrite = TRUE)
#     Convert("seurat_integrated.h5Seurat", dest = "h5ad",overwrite = TRUE)
#     """
#     # print(r(rscript))
#     adata = sc.read_h5ad('./seurat_integrated.h5ad')
#     return adata

def integrate_prepare(adata_whole):
    """
    一种特殊的hvg，在src.base中中已经产生adata_whole
    1，mapping 1v1 gene到ref
    2，各个数据集分别hvg
    3，各个数据集取交
    :param adata:
    :return:
    """
    adata_raw = sc.AnnData(X=adata_whole.raw.X, obs=adata_whole.obs, var=adata_whole.raw.var, uns=adata_whole.uns)
    adata_hvg = adata_whole
    return adata_raw, adata_hvg


def integrate(adata_whole, method, batch_key='batch'):
    adata_raw, adata_hvg = integrate_prepare(adata_whole.copy())

    if method == 'raw':  # raw or combat
        adata_int = adata_hvg
    elif method == 'combat':
        sc.pp.combat(adata_hvg, key='batch')
        adata_int = adata_hvg
    elif method == 'cell_train_class':  # 推荐
        adata_hvg.obsm['X_emb'] = adata_hvg.obsm['cell_train_class']
        adata_int = adata_hvg
    elif method == 'cell_train_hidden':
        adata_hvg.obsm['X_emb'] = adata_hvg.obsm['cell_train_hidden']
        adata_int = adata_hvg
    elif method == 'cell_train_hidden_eval':
        adata_hvg.obsm['X_emb'] = adata_hvg.obsm['cell_train_hidden_eval']
        adata_int = adata_hvg
    elif method == 'cell_pretrain_hidden':
        adata_hvg.obsm['X_emb'] = adata_hvg.obsm['cell_pretrain_hidden']
        adata_int = adata_hvg
    elif method == 'scanorama':  # scanorama可以imputation也可以embedding
        # https://scanpy.readthedocs.io/en/stable/external.html
        sce.pp.scanorama_integrate(adata_hvg, key='batch')
        adata_hvg.obsm['X_emb'] = adata_hvg.obsm['X_scanorama']  # 以scib为标准
        adata_int = adata_hvg
    elif method == 'harmony':
        # https://scanpy.readthedocs.io/en/stable/external.html
        sc.tl.pca(adata_hvg)
        sce.pp.harmony_integrate(adata_hvg, key=batch_key)
        adata_hvg.obsm['X_emb'] = adata_hvg.obsm['X_pca_harmony']  # 以scib为标准
        adata_int = adata_hvg
    elif method == 'pyliger':
        # https://github.com/welch-lab/pyliger/blob/master/integrating_multi_scRNA_data.ipynb
        adata_int = pyliger_integrate(adata_raw)
    elif method == 'scvi':
        # ref https://github.com/theislab/scib-pipeline/blob/main/scripts/integration/runIntegration.py
        # integrate
        adata_int, vae = scvi_integrate(adata_hvg)
    elif method == 'scalex':
        from scalex import SCALEX
        condition = adata_raw.obs.loc[:, 'batch'].unique().tolist()
        data_list = [adata_raw[adata_raw.obs.loc[:, 'batch'] == item, :].copy() for item in condition]
        adata_int = SCALEX(data_list, condition, min_features=1, min_cells=1, ignore_umap=True, max_iteration=30000)
        adata_int.obsm['X_emb'] = adata_int.obsm['latent']
    elif method == 'seurat':  # TODO
        # 学习python对r的调用或者直接在r中run
        adata_int = seurat_integrate(adata_int)
    elif method == 'bbknn':  # not need
        # https://scanpy-tutorials.readthedocs.io/en/latest/integrating-data-using-ingest.html
        sc.external.pp.bbknn(adata_hvg, batch_key='batch')
        adata_int = adata_hvg
    else:
        raise NotImplementedError

    return adata_raw, adata_int


def clear_fig(fig):
    if fig:
        fig.axes[0].set_xlabel(None)
        fig.axes[0].set_ylabel(None)
        fig.tight_layout()
    else:
        pass
    return fig


def visualize(adata_int, method, batch_key='batch', cell_key='cell_ontology_class', subcell_key=None, legend_loc='right margin', dis=0.5):
    # 选择算法，计算邻居
    if method in ['raw', 'combat']:
        sc.tl.pca(adata_int, n_comps=30)
        sc.pp.neighbors(adata_int, use_rep='X_pca')
    elif method in ['scanorama', 'harmony', 'pyliger', 'scvi', 'scanvi', 'scalex',
                    'cell_train_class', 'cell_train_hidden', 'cell_pretrain_hidden', 'cell_train_hidden_eval']:
        sc.pp.neighbors(adata_int, use_rep='X_emb', n_neighbors=15, metric='cosine')
    elif method in ['bbknn']:
        pass  # bbknn修正了neighbors和connectivity
    elif method in ['seurat']:
        pass
    else:
        raise NotImplementedError

    # leiden
    sc.tl.leiden(adata_int, resolution=0.5)

    # 绘图准备
    sc.settings.set_figure_params(dpi=300, facecolor='white', dpi_save=300, figsize=(4, 4))
    # umap
    sc.tl.umap(adata_int, min_dist=dis)

    # # vis
    # fig_cell = sc.pl.umap(adata_int, color=[cell_key], return_fig=True, title='')
    # # fig_cell = plot_scatter(adata_int, fig_cell)  # TODO 是否画点
    # fig_batch = sc.pl.umap(adata_int, color=[batch_key], return_fig=True, title='')
    # fig_leiden = sc.pl.umap(adata_int, color=['leiden'], return_fig=True, title='')
    # if subcell_key is None:
    #     fig_subcell = None
    # else:
    #     fig_subcell = sc.pl.umap(adata_int, color=subcell_key, return_fig=True, title='')

    # # trajectory
    # sc.tl.paga(adata_int, groups='leiden')
    # sc.pl.paga(adata_int, color=['leiden'])
    # sc.tl.paga(adata_int, groups='cell_ontology_class')
    # sc.pl.paga(adata_int, color=['cell_ontology_class'])
    # sc.tl.paga(adata_int, groups='cell_type')
    # sc.pl.paga(adata_int, color=['cell_type'])

    # vis
    fig_cell = sc.pl.umap(adata_int, color=[cell_key], return_fig=True, legend_loc=legend_loc, title='')
    # fig_cell = plot_scatter(adata_int, fig_cell)  # TODO 是否画点
    fig_batch = sc.pl.umap(adata_int, color=[batch_key], return_fig=True, legend_loc=legend_loc, title='')
    fig_leiden = sc.pl.umap(adata_int, color=['leiden'], return_fig=True, legend_loc=legend_loc, title='')
    if subcell_key is None:
        fig_subcell = None
    else:
        fig_subcell = sc.pl.umap(adata_int, color=subcell_key, return_fig=True, legend_loc=legend_loc, title='')

    figs = [fig_cell, fig_batch, fig_leiden, fig_subcell]
    if method in ['raw', 'combat']:
        embedding_corr_dict = cal_diff(adata_int, emb='X_pca')
    elif method in ['scanorama', 'harmony', 'pyliger', 'scvi', 'scanvi', 'scalex',
                    'cell_train_class', 'cell_train_hidden', 'cell_pretrain_hidden', 'cell_train_hidden_eval']:
        embedding_corr_dict = cal_diff(adata_int, emb='X_emb')
    # embedding_corr_dict = cal_diff(adata_int, emb='X_pca')
    else:
        raise NotImplementedError

    # return figs, embedding_corr_dict
    return [clear_fig(fig) for fig in figs], embedding_corr_dict


def plot_scatter(adata, fig):
    """
    无法确定折线图的每个顶点，所以绘制散点图
    :param adata:
    :param fig:
    :return:
    """
    batches = adata.obs.loc[:, 'batch'].unique()
    for batch in batches:
        # 获取当前批次数据
        adata_temp = adata[adata.obs.loc[:, 'batch'] == batch, :]
        # 当前批次的细胞类型
        cell_types = adata_temp.obs.loc[:, 'cell_ontology_class'].unique()
        umap_list = []
        for cell_type in cell_types:
            temp = adata_temp[adata_temp.obs.loc[:, 'cell_ontology_class'] == cell_type, :]
            umap = np.mean(temp.obsm['X_umap'], axis=0).tolist()
            umap_list.append(umap)
        # 绘制折线图
        umap_array = np.array(umap_list)
        fig.axes[0].scatter(umap_array[:, 0], umap_array[:, 1], color='black', marker='o')
    return fig


def cal_diff(adata, emb='X_emb'):
    """
    计算umap和embedding的距离
    :param adata:
    :param fig:
    :return:
    """
    umap_corr_dict = {}
    embedding_corr_dict = {}
    batches = adata.obs.loc[:, 'batch'].unique()
    cell_types = adata.obs.loc[:, 'cell_ontology_class'].unique()
    for cell_type in cell_types:
        # 获取当前批次数据
        adata_temp = adata[adata.obs.loc[:, 'cell_ontology_class'] == cell_type, :]
        # 当前批次的细胞类型
        umap_list = []
        embedding_list = []
        for batch in batches:
            temp = adata_temp[adata_temp.obs.loc[:, 'batch'] == batch, :]
            umap = np.mean(temp.obsm['X_umap'], axis=0).tolist()    # TODO
            umap_list.append(umap)
            embedding = np.mean(temp.obsm[emb], axis=0).tolist()    # TODO
            embedding_list.append(embedding)
        # 绘制当前cell_type的corr
        umap_array = np.array(umap_list)
        embedding_array = np.array(embedding_list)

        umap_corr = np.nan_to_num(np.corrcoef(umap_array))  # TODO umap用什么计算距离
        embedding_corr = np.nan_to_num(np.corrcoef(embedding_array))  # 去掉nan

        umap_corr_df = pd.DataFrame(data=umap_corr, columns=batches, index=batches)
        embedding_corr_df = pd.DataFrame(data=embedding_corr, columns=batches, index=batches)
        # append
        umap_corr_dict[cell_type] = umap_corr_df
        embedding_corr_dict[cell_type] = embedding_corr_df

    # mean
    umap_corr_dict['mean'] = sum([data for data in umap_corr_dict.values()]) / len(umap_corr_dict)  # TODO 存在未知细胞类型时会出错
    embedding_corr_dict['mean'] = sum([data for data in embedding_corr_dict.values()]) / len(embedding_corr_dict)

    return embedding_corr_dict
