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
    {'preprocess':  # We only recommend modifying data param in preprocess.
        {
            'path': './dataset/',
            'dataset_file': pd.DataFrame(
                # 'specie1 dataset', 'True represents specie1 dataset has the manual annotation, and vice versa does not', 'many-to-many homologous genes', 'specie2 dataset', 'True represents specie2 dataset has the manual annotation, and vice versa does not'.

                [['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2monkey.csv',
                  'raw-liver-monkey-Martin.h5ad', False],

                 ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2mouse.csv',
                  'raw-liver-mouse-Martin.h5ad', False],

                 ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2zebrafish.csv',
                  'raw-liver-zebrafish-ggj5.h5ad', False],

                 ],
                columns=['source', 'source label', 'relationship', 'destination', 'destination label']),  # column names indicate the above files
            'graph_mode': 'undirected',
            'feature_gene': 'HIG',
            'sample_ratio': 1,  # default 1, set to ratio of (0, 1] to down sample the dataset
            'get_balance': 'False'  # set ref and query with the same cell type
        },

        'train': {  # We only recommend modifying device and train_mode params in train.
            'device': 'cuda:0',  # cpu or cuda
            'train_mode': 'mini_batch',  # mini_batch or full batch
            'dim_hidden': 128,
            'batch_size': 1024,
            'gnn_layer_num': 2,
            'epoch_integration': 10,
            'epoch_annotation': 10,
            'encoder': 'GCN',
            'classifier': 'GAT',
            'res': True,
            'share': True,
            'cluster': False,
            'epoch_cluster': 10,
            'cluster_num': 5,
            'domain': False,
            'reconstruct': True,

        },

        'postprocess': {}
    }
