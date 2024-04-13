# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/7/17 13:54
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : params.py
# @IDE     : PyCharm

import pandas as pd

# TODO yaml
PARAMS = \
    {'preprocess':
        {
            # We only recommend modifying the following 2 hyperparameter including path and dataset_file.
            'path': './dataset/',
            'dataset_file': pd.DataFrame(
                [['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2monkey.csv',
                  'raw-liver-monkey-Martin.h5ad', False],

                 ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2mouse.csv',
                  'raw-liver-mouse-Martin.h5ad', False],

                 ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2zebrafish.csv',
                  'raw-liver-zebrafish-ggj5.h5ad', False],

                 ],
                # 'specie1 dataset', 'True represents specie1 dataset has the manual annotations, and vice versa does not', 'many-to-many homologous genes', 'specie2 dataset', 'True represents specie2 dataset has the manual annotations, and vice versa does not'.
                columns=['source', 'source label', 'relationship', 'destination', 'destination label']),  # column names indicate the above files

            # do not need to change
            'graph_mode': 'undirected',  # undirected or directed, default undirected
            'feature_gene': 'HIG',  # feature type
            'sample_ratio': 1,  # default 1, set to ratio of (0, 1] to down sample the dataset
            'get_balance': 'False'  # set ref and query with the same cell type
        },

        'train': {
            # We only recommend modifying the following 5 hyperparameter or use the default value.
            'device': 'cuda:0',  # cpu or cuda
            'train_mode': 'mini_batch',  # mini_batch or full batch
            'epoch_integration': 10,   # integration epoch
            'epoch_annotation': 10,  # annotation epoch
            'batch_size': 1024,  # batch_size

            # do not need to change
            'dim_hidden': 128,  # the dims of cell or gene embedding
            'gnn_layer_num': 2,  # the number of gnn layers
            'encoder': 'GCN',   # the type of gnn encoder
            'classifier': 'GAT',  # the type of classifier encoder
            'res': True,   # use residual or not
            'share': True,   # share the parameters or not
            'cluster': False,   # the epoch of clusters in training step
            'epoch_cluster': 10,   # the number of epochs for clustering in training step
            'cluster_num': 5,  # the number of clusters in training step
            'domain': False,  # use domain adaption or not
            'reconstruct': True,  # reconstruct the node feature or not

        },

        'postprocess': {}
    }
