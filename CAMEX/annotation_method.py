# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/9/4 22:18
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : annotation_method.py
# @IDE     : PyCharm

import numpy as np
import scanpy as sc
import pandas as pd
import torch
import os

from torch import nn
from CAMEX.annotation_metrics import evaluate_all


"""
：+
：
"""


def cal_came2(adata, method):
    """

    :param adata:
    :param method:
    :return:
    """
    # split adata into train and test
    batch = list(adata.uns['dataset_type'].keys())

    # cell_type_raw
    cell_ontology_dict = {num: name for name, num in adata.uns['cell_type'].items()}

    # cell_type_unify
    cell_class_temp = adata.obs.loc[:, ['cell_class', 'cell_class_num']].to_numpy()
    cell_class_dict = {item[1]: item[0] for item in cell_class_temp}
    if 'unknown' not in cell_class_dict.values():
        cell_class_dict[len(cell_class_dict)] = 'unknown'  #

    # evaluate
    results = pd.DataFrame()
    confusion_matrix_unify = {}
    confusion_matrix_raw = {}
    adata_list = []
    for i, dataset_name in enumerate(batch):
        #
        adata_temp = adata[adata.obs.loc[:, 'batch'] == batch[i], :]

        #
        y_ontology = adata_temp.obs['cell_ontology_class_num'].to_numpy()

        # , y_pred, y_prob
        y_true = adata_temp.obs['cell_class_num'].to_numpy()
        y_prob = nn.Softmax(dim=-1)(torch.tensor(adata_temp.obsm['cell_train_class'])).numpy()  # CE自带SF，现在添加
        y_pred = np.argmax(y_prob, axis=-1)
        #
        adata_temp.obs.loc[:, 'pred'] = y_pred
        adata_temp.obs.loc[:, 'pred_name'] = adata_temp.obs.loc[:, 'pred'].map(cell_class_dict)
        adata_list.append(adata_temp)

        #
        y_ontology = np.concatenate([y_ontology, np.arange(y_prob.shape[1], dtype=np.int32)])
        y_true = np.concatenate([y_true, np.arange(y_prob.shape[1], dtype=np.int32)])
        y_prob = np.concatenate([y_prob, np.eye(y_prob.shape[1], dtype=np.float32)])
        y_pred = np.concatenate([y_pred, np.arange(y_prob.shape[1], dtype=np.int32)])

        # evaluate_metrics
        cm_unify, cm_raw, result_metrics = evaluate_all(y_true, y_prob, y_pred, y_ontology, cell_ontology_dict,
                                                        cell_class_dict, method, dataset_name)

        # save
        confusion_matrix_unify['dataset_name'] = cm_unify
        confusion_matrix_raw['dataset_name'] = cm_raw
        result_metrics = pd.DataFrame.from_dict(result_metrics, orient="index")
        result_metrics.columns = [method + '-' + dataset_name]
        results = pd.concat([results, result_metrics], axis=1)
        print()

    # save embedding
    adata = sc.concat(adata_list)
    adata.uns['confusion_matrix_unify'] = confusion_matrix_unify
    adata.uns['confusion_matrix_raw'] = confusion_matrix_raw
    return adata, results


def evaluate_cellblast(model, adata_test):
    # ref_cell_type list
    ref_cell_type = model.blast.ref.get_meta_or_var(['cell_class_num'], normalize_var=False,
                                                    log_var=True).values.ravel()
    # cell type dict
    cell_type = list(np.unique(ref_cell_type))
    cell_type.append(len(cell_type))    # unknown

    # count matrix
    vote_matrix = pd.DataFrame(data=np.zeros((len(adata_test.obs), len(cell_type))), columns=cell_type,
                               index=adata_test.obs.index)
    prob_matrix = pd.DataFrame(data=np.zeros((len(adata_test.obs), len(cell_type))), columns=cell_type,
                               index=adata_test.obs.index)

    for i, _hits in enumerate(model.hits):
        hits = ref_cell_type[_hits]

        #
        if hits.size <= 1:
            hits = np.array(cell_type)  #

        label, count = np.unique(hits, return_counts=True)  #
        best_idx = count.argmax()
        if count[best_idx] / hits.size <= 0.2:
            # ，
            vote_matrix.loc[vote_matrix.index[i], len(cell_type)-1] = 1
            prob_matrix.loc[prob_matrix.index[i], len(cell_type)-1] = 1
            continue
        else:
            #
            for j, item in enumerate(label):
                vote_matrix.loc[vote_matrix.index[i], item] = count[j]
                prob_matrix.loc[prob_matrix.index[i], item] = count[j] / sum(count)
        # prob_matrix.iloc[i, :] = torch.nn.Softmax()(torch.tensor(count_matrix.iloc[i, :].to_numpy())).numpy()

    return vote_matrix, prob_matrix


def anno_prob(model,
              adata_test,
              min_hits: int = 2,
              majority_threshold: float = 0.5,
              ):
    ref_cell_type = model.blast.ref.obs['cell_class_num'].to_numpy().ravel()
    # cell type dict
    cell_type = list(np.unique(ref_cell_type))
    cell_type.append(len(cell_type))
    # count matrix
    vote_matrix = pd.DataFrame(data=np.zeros((len(adata_test.obs), len(cell_type))), columns=cell_type,
                               index=adata_test.obs.index)
    prob_matrix = pd.DataFrame(data=np.zeros((len(adata_test.obs), len(cell_type))), columns=cell_type,
                               index=adata_test.obs.index)
    # 填补
    for i, _hits in enumerate(model.hits):
        hits = ref_cell_type[_hits.astype(int)]
        #
        if hits.size < min_hits:
            hits = np.array(cell_type)  #
        #
        label, count = np.unique(hits, return_counts=True)
        best_idx = count.argmax()
        if count[best_idx] / hits.size <= majority_threshold:
            #
            vote_matrix.loc[vote_matrix.index[i], len(cell_type) - 1] = 1
            prob_matrix.loc[prob_matrix.index[i], len(cell_type) - 1] = 1
            continue
        else:
            #
            for j, item in enumerate(label):
                vote_matrix.loc[vote_matrix.index[i], item] = count[j]
                prob_matrix.loc[prob_matrix.index[i], item] = count[j] / sum(count)
        # softmax
        # prob_matrix.iloc[i, :] = torch.nn.Softmax()(torch.tensor(count_matrix.iloc[i, :].to_numpy())).numpy()

    return vote_matrix, prob_matrix


def cal_cellblast(adata, method):
    import Cell_BLAST as cb
    # from Cell_BLAST.data import ExprDataSet

    # cell_type_raw
    cell_ontology_dict = {num: name for name, num in adata.uns['cell_type'].items()}

    # cell_type_unify
    cell_class_temp = adata.obs.loc[:, ['cell_class', 'cell_class_num']].to_numpy()
    cell_class_dict = {item[1]: item[0] for item in cell_class_temp}
    if 'unknown' not in cell_class_dict.values():
        cell_class_dict[len(cell_class_dict)] = 'unknown'  #

    # split adata into train and test
    batch = adata.obs.loc[:, 'batch'].unique().to_numpy()
    adata_train = adata[adata.obs.loc[:, 'batch'] == batch[0], :]
    adata_dict = {}
    for i in range(0, len(batch)):
        adata_dict[batch[i]] = adata[adata.obs.loc[:, 'batch'] == batch[i], :]

    # train
    models = []
    # for i in range(2):
    #     models.append(cb.directi.fit_DIRECTi(adata_train, latent_dim=10, epoch=10, genes=adata_train.uns['hvg']))
    for i in range(2):
        models.append(cb.directi.fit_DIRECTi(
            adata_train, genes=adata_train.uns['hvg'], epoch=1000,
            latent_dim=10, cat_dim=20, random_seed=i
        ))
    blast = cb.blast.BLAST(models, adata_train)
    # # save and load
    # blast.save("./baron_human_blast")
    # blast = cb.blast.BLAST.load("./baron_human_blast")

    # cell_type_unify
    cell_class_temp = adata.obs.loc[:, ['cell_class', 'cell_class_num']].to_numpy()
    cell_class_dict = {item[1]: item[0] for item in cell_class_temp}
    if 'unknown' not in cell_class_dict.values():
        cell_class_dict[len(cell_class_dict)] = 'unknown'  #

    # predict
    adata_list = []
    results = pd.DataFrame()
    confusion_matrix_unify = {}
    confusion_matrix_raw = {}
    X_emb = None
    for dataset_name, adata_temp in adata_dict.items():
        # save embedding
        if X_emb is None:
            X_emb = models[0].inference(adata_temp)
        else:
            X_emb = np.concatenate([X_emb, models[0].inference(adata_temp)])

        # predict by multiple model
        hits = blast.query(adata_temp)
        # vote
        hits = hits.reconcile_models().filter(by="pval", cutoff=0.05)
        # official predict
        official_predictions = hits.annotate("cell_class")  # TODO
        # my predict
        vote_matrix, prob_matrix = anno_prob(hits, adata_temp)
        y_true = adata_temp.obs.loc[:, 'cell_class_num'].to_numpy()
        y_prob = prob_matrix.to_numpy()
        y_pred = np.argmax(y_prob, axis=-1)

        # save
        adata_temp.obs.loc[:, 'pred'] = y_pred
        adata_temp.obs.loc[:, 'pred_name'] = adata_temp.obs.loc[:, 'pred'].map(cell_class_dict)  #
        adata_list.append(adata_temp)

        #
        y_ontology = adata_temp.obs['cell_ontology_class_num'].to_numpy()

        #
        y_ontology = np.concatenate([y_ontology, np.arange(y_prob.shape[1], dtype=np.int32)])
        y_true = np.concatenate([y_true, np.arange(y_prob.shape[1], dtype=np.int32)])
        y_prob = np.concatenate([y_prob, np.eye(y_prob.shape[1], dtype=np.float32)])
        y_pred = np.concatenate([y_pred, np.arange(y_prob.shape[1], dtype=np.int32)])

        # metrics
        cm_unify, cm_raw, result_metrics = evaluate_all(y_true, y_prob, y_pred, y_ontology, cell_ontology_dict,
                                                        cell_class_dict, method, dataset_name)

        # save
        confusion_matrix_unify['dataset_name'] = cm_unify
        confusion_matrix_raw['dataset_name'] = cm_raw
        result_metrics = pd.DataFrame.from_dict(result_metrics, orient="index")
        result_metrics.columns = [method + '-' + dataset_name]
        results = pd.concat([results, result_metrics], axis=1)

    # adata_whole = sc.concat(adata_list)
    adata_whole = sc.concat(adata_list)
    adata_whole.uns['confusion_matrix_unify'] = confusion_matrix_unify
    adata_whole.uns['confusion_matrix_raw'] = confusion_matrix_raw

    # save embedding
    adata_whole.obsm['X_emb'] = X_emb
    return adata_whole, results


def cal_singlecellnet(adata, method):
    # cell_type_raw
    cell_ontology_dict = {num: name for name, num in adata.uns['cell_type'].items()}
    # cell_type_unify
    cell_class_temp = adata.obs.loc[:, ['cell_class', 'cell_class_num']].to_numpy()
    cell_class_dict = {item[1]: item[0] for item in cell_class_temp}
    if 'unknown' not in cell_class_dict.values():
        cell_class_dict[len(cell_class_dict)] = 'unknown'  #

    import pySingleCellNet as pySCN
    # split to train and test
    batch = adata.obs.loc[:, 'batch'].unique().to_numpy()
    adata_train = adata[adata.obs.loc[:, 'batch'] == batch[0], :]
    adata_dict = {}
    for i in range(0, len(batch)):
        adata_dict[batch[i]] = adata[adata.obs.loc[:, 'batch'] == batch[i], :]

    # train
    [gene1, gene2, model] = pySCN.scn_train(adata_train, nTopGenes=100, nRand=100, nTrees=1000, nTopGenePairs=100,
                                            dLevel="cell_class_num", stratify=True, limitToHVG=True)
    # save
    adata_list = []
    results = pd.DataFrame()
    confusion_matrix_unify = {}
    confusion_matrix_raw = {}
    X_emb = None
    for dataset_name, adata_temp in adata_dict.items():
        adata_pred = pySCN.scn_classify(adata_temp, gene1, gene2, model, nrand=0)

        # my predict
        y_true = adata_temp.obs['cell_class_num'].to_numpy()
        y_prob = adata_pred.X
        adata_pred.obs['SCN_class'] = adata_pred.obs['SCN_class'].apply(lambda x: len(cell_class_dict) - 1 if x == 'rand' else x)
        y_pred = adata_pred.obs['SCN_class'].astype(np.int).to_numpy()
        adata_temp.obs.loc[:, 'pred'] = y_pred
        adata_temp.obs.loc[:, 'pred_name'] = adata_temp.obs.loc[:, 'pred'].map(cell_class_dict)
        y_ontology = adata_temp.obs['cell_class_num']

        #
        y_ontology = np.concatenate([y_ontology, np.arange(y_prob.shape[1], dtype=np.int32)])
        y_true = np.concatenate([y_true, np.arange(y_prob.shape[1], dtype=np.int32)])
        y_prob = np.concatenate([y_prob, np.eye(y_prob.shape[1], dtype=np.float32)])
        y_pred = np.concatenate([y_pred, np.arange(y_prob.shape[1], dtype=np.int32)])

        # metrics
        cm_unify, cm_raw, result_metrics = evaluate_all(y_true, y_prob, y_pred, y_ontology, cell_ontology_dict,
                                                        cell_class_dict, method, dataset_name)

        # save embedding
        if X_emb is None:
            X_emb = y_prob[: len(adata_temp)]
        else:
            X_emb = np.concatenate([X_emb, y_prob[: len(adata_temp)]])
        adata_list.append(adata_temp)
        confusion_matrix_unify[dataset_name] = cm_unify
        confusion_matrix_raw[dataset_name] = cm_raw
        result_metrics = pd.DataFrame.from_dict(result_metrics, orient="index")
        result_metrics.columns = [method + '-' + dataset_name]
        results = pd.concat([results, result_metrics], axis=1)

    # save embedding
    adata_whole = sc.concat(adata_list)
    adata_whole.obsm['X_emb'] = X_emb
    adata_whole.uns['confusion_matrix_unify'] = confusion_matrix_unify
    adata_whole.uns['confusion_matrix_raw'] = confusion_matrix_raw
    return adata_whole, results


def cal_scvi(adata, method):
    # cell_type_raw
    cell_ontology_dict = {num: name for name, num in adata.uns['cell_type'].items()}
    # cell_type_unify
    cell_class_temp = adata.obs.loc[:, ['cell_class', 'cell_class_num']].to_numpy()
    cell_class_dict = {item[1]: item[0] for item in cell_class_temp}
    if 'unknown' not in cell_class_dict.values():
        cell_class_dict[len(cell_class_dict)] = 'unknown'  #

    import scvi
    # device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ，
    batch = adata.obs.loc[:, 'batch'].unique().to_numpy()
    adata_train = adata[adata.obs.loc[:, 'batch'] == batch[0], :]
    adata_train.obs.loc[:, 'scvi_cell_type'] = adata_train.obs.loc[:, 'cell_class_num'].astype(str)
    adata_test_list = []
    # ，，
    for i in range(1, len(batch)):
        adata_temp = adata[adata.obs.loc[:, 'batch'] == batch[i], :]
        adata_temp.obs.loc[:, 'scvi_cell_type'] = 'unknown'
        adata_test_list.append(adata_temp)
    adata_test_list.insert(0, adata_train)
    adata = sc.concat(adata_test_list)

    # pretrain
    scvi.model.SCVI.setup_anndata(adata, batch_key='batch', labels_key="scvi_cell_type")
    scvi_model = scvi.model.SCVI(adata)
    scvi_model.train(max_epochs=100, use_gpu=device)
    # train
    scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model, 'unknown')
    scanvi_model.train(max_epochs=100, use_gpu=device)

    # get label and representation
    adata.obs["class_scANVI"] = scanvi_model.predict(adata)  # 返回预测值
    adata.obs = pd.concat([adata.obs, scanvi_model.predict(adata, soft=True)], axis=1)
    adata.uns['predict_prob'] = scanvi_model.predict(adata, soft=True)
    # TODO embedding
    adata.obsm["X_emb"] = scanvi_model.get_latent_representation(adata)  # embedding

    # cell_type
    cell_type_scvi = adata.uns['predict_prob'].columns.to_numpy()

    # save
    adata_list = []
    results = pd.DataFrame()
    confusion_matrix_unify = {}
    confusion_matrix_raw = {}
    for dataset_name in batch:  #
        adata_temp = adata[adata.obs.loc[:, 'batch'] == dataset_name]
        # adata_list.append(adata_temp)
        # my predict
        y_true = adata_temp.obs['cell_class_num'].to_numpy()
        y_prob_df = adata_temp.obs.loc[:, cell_type_scvi]
        y_prob_df['unknown'] = 0
        y_prob = y_prob_df.to_numpy()
        y_pred = adata_temp.obs['class_scANVI'].to_numpy().astype(int)
        adata_temp.obs.loc[:, 'pred'] = y_pred
        adata_temp.obs.loc[:, 'pred_name'] = adata_temp.obs.loc[:, 'pred'].map(cell_class_dict)

        #
        y_ontology = adata_temp.obs['cell_ontology_class_num'].to_numpy()

        #
        y_ontology = np.concatenate([y_ontology, np.arange(y_prob.shape[1], dtype=np.int32)])
        y_true = np.concatenate([y_true, np.arange(y_prob.shape[1], dtype=np.int32)])
        y_prob = np.concatenate([y_prob, np.eye(y_prob.shape[1], dtype=np.float32)])
        y_pred = np.concatenate([y_pred, np.arange(y_prob.shape[1], dtype=np.int32)])

        # metrics
        cm_unify, cm_raw, result_metrics = evaluate_all(y_true, y_prob, y_pred, y_ontology, cell_ontology_dict,
                                                        cell_class_dict, method, dataset_name)

        # save
        adata_list.append(adata_temp)
        confusion_matrix_unify[dataset_name] = cm_unify
        confusion_matrix_raw[dataset_name] = cm_raw
        result_metrics = pd.DataFrame.from_dict(result_metrics, orient="index")
        result_metrics.columns = [method + '-' + dataset_name]
        results = pd.concat([results, result_metrics], axis=1)

    # save embedding
    adata = sc.concat(adata_list)
    # adata_whole = adata_list[0]
    # for i in range(1, len(adata_list)):
    #     adata_whole = adata_whole.concatenate(adata_list[i])
    return adata, results


def cal_itclust(adata, method):
    import ItClust as ic
    #
    adata.obs.loc[:, 'celltype'] = adata.obs.loc[:, 'cell_ontology_class']

    # split adata into train and test
    batch = adata.obs.loc[:, 'batch'].unique().to_numpy()
    adata_train = adata[adata.obs.loc[:, 'batch'] == batch[0], :]
    adata_test_dict = {}
    for i in range(len(batch)):
        adata_test_dict[batch[i]] = adata[adata.obs.loc[:, 'batch'] == batch[i], :]

    # train，，
    clf = ic.transfer_learning_clf()
    clf.fit(adata_train, adata, pretrain_epochs=100, maxiter=100)
    pred, prob, cell_type_pred = clf.predict()
    #
    prob['unknown'] = 0

    # test
    # TODO ，，
    cell_type_dict = {v[0]: int(k) for k, v in cell_type_pred.items()}
    cell_type_dict['unknown'] = len(cell_type_dict)
    results = pd.DataFrame()
    confusion_matrix_dict = {}
    for name, data in adata_test_dict.items():
        y_true = data.obs.loc[:, 'cell_ontology_class'].apply(
            lambda x: cell_type_dict[x] if x in cell_type_dict.keys() else cell_type_dict['unknown']).to_numpy()
        #
        y_true = np.concatenate([y_true, np.arange(0, len(cell_type_dict))])
        # target
        prob_temp = prob.loc[data.obs.index + '-target', :]
        y_pred_prob = prob_temp.to_numpy()
        y_pred_prob = np.concatenate([y_pred_prob, np.eye(len(cell_type_dict))])
        y_pred = np.argmax(y_pred_prob, axis=-1)

        # evaluate
        result, cm = evaluate_all(method + "_" + name, y_true, y_pred, y_pred_prob)
        results = pd.concat([results, result], axis=1)
        confusion_matrix_dict[name] = cm

    # save embedding，
    clf.adata_test.obsm['X_emb'] = clf.adata_test.obsm['X_Embeded_zisy_trans_True']

    adata = clf.adata_test
    adata.uns['confusion_matrix_dict'] = confusion_matrix_dict
    return adata, results


def run_seurat():
    from rpy2.robjects import r
    rscript = """
    rm(list=ls())
    library(Seurat)
    library(SeuratDisk)
    library(zellkonverter)
    # 
    print('load data')
    file_name <- read.table('adata_file.csv')[, 1]
    adata_list <- list()
    for (i in 1:length(file_name)) {
      print(file_name[[i]])
      data_temp <- readH5AD(file_name[[i]])
      adata_temp <- as.Seurat(data_temp, counts = "X", data = NULL)
      adata_list[i] <- adata_temp
    }

    # 
    print('preprocess')
    for (i in 1:length(adata_list)) {
      adata_list[[i]] <- NormalizeData(adata_list[[i]], verbose = FALSE)
      adata_list[[i]] <- FindVariableFeatures(adata_list[[i]], selection.method = "vst", nfeatures = 2000,
                                              verbose = FALSE)
      adata_list[[i]] <- ScaleData(adata_list[[i]])
      adata_list[[i]] <- RunPCA(adata_list[[i]], npcs = 30, verbose = FALSE)
    }

    # ref and query
    ref <- adata_list[[1]]
    query_list <- adata_list[1: length(adata_list)]

    # 
    for (i in 1:length(query_list)) {
      anchors <- FindTransferAnchors(reference = ref, query = query_list[[i]],
                                     dims = 1:30, reference.reduction = "pca")
      # 
      predictions <- TransferData(anchorset = anchors, refdata = ref$cell_ontology_class,
                                  dims = 1:30)
      # 
      query_list[[i]] <- AddMetaData(query_list[[i]], metadata = predictions)
      # 
      query_list[[i]]$prediction.match <- query_list[[i]]$predicted.id == query_list[[i]]$cell_ontology_class
      table(query_list[[i]]$prediction.match)
      table(query_list[[i]]$predicted.id)
      # 
      library(SeuratDisk)
      SaveH5Seurat(query_list[[i]], filename = paste('seurat_annotation_',strsplit(file_name[i], '.', fixed = T)[[1]][1],'.h5Seurat',sep = ""),overwrite = TRUE)
      Convert(paste('seurat_annotation_',strsplit(file_name[i], '.', fixed = T)[[1]][1],'.h5Seurat',sep = ""), dest = "h5ad",overwrite = TRUE)
    }
    """
    print(r(rscript))


def cal_seurat(adata, method):
    # mkdir
    try:
        os.mkdir('./seurat')
    except:
        print('seurat file has been created')
        pass

    # save h5ad
    batch = adata.obs.loc[:, 'batch'].unique()
    batch_df = pd.DataFrame(batch)
    batch_df.iloc[:, 0] = batch_df.iloc[:, 0].apply(lambda x: x + '.h5ad')
    batch_df.to_csv('./adata_file.csv', index=False, header=False)
    for name in batch:
        adata_temp = adata[adata.obs.loc[:, 'batch'] == name, :]
        adata_temp.obs['cell_ontology_backup'] = adata_temp.obs['cell_ontology_class']
        adata_temp.write_h5ad('./' + name + '.h5ad')

    # # run seurat TODO
    # run_seurat()

    # load seurat file, evaluate and save result
    # adata_train
    adata_train = adata[adata.obs.loc[:, 'batch'] == batch[0], :]
    cell_type_num = len(adata_train.obs['cell_ontology_class'].unique().tolist())
    adata_list = []
    results = pd.DataFrame()
    confusion_matrix_dict = {}
    for name in batch:
        # load annotation data
        adata_temp = sc.read_h5ad('seurat_annotation_' + name + '.h5ad')
        # adata_temp ontology class
        adata_temp.obs['cell_ontology_class'] = adata[adata_temp.obs.index,].obs['cell_ontology_class']
        # ，
        cell_type = adata_temp.obs.columns[-2 - cell_type_num:-2].tolist()  #
        cell_type = [' '.join(item.split('.')[2:]) for item in cell_type]
        cell_type_dict = {cell_type[i]: i for i in range(len(cell_type))}
        cell_type_dict['unknown'] = len(cell_type_dict)
        # TODO
        y_true = adata_temp.obs['cell_ontology_class'].apply(
            lambda x: cell_type_dict[x] if x in cell_type_dict.keys() else cell_type_dict['unknown']).to_numpy()
        y_true = np.concatenate([y_true, np.arange(0, len(cell_type_dict))])
        y_pred_prob = adata_temp.obs.iloc[:, -2 - cell_type_num:-2]
        y_pred_prob['unknown'] = 0
        y_pred_prob = y_pred_prob.to_numpy()
        y_pred_prob = np.concatenate([y_pred_prob, np.eye(len(cell_type_dict))])
        y_pred = np.argmax(y_pred_prob, axis=-1)

        # evaluate
        result, cm = evaluate_all(method + "_" + name, y_true, y_pred, y_pred_prob)
        results = pd.concat([results, result], axis=1)
        confusion_matrix_dict[name] = cm
        adata_list.append(adata_temp)

    # embedding，
    adata = sc.concat(adata_list)
    sc.tl.pca(adata)
    adata.obsm['X_emb'] = adata.obsm['X_pca']
    adata.uns['confusion_matrix_dict'] = confusion_matrix_dict
    return adata, results


if __name__ == '__main__':
    print('hello world')
