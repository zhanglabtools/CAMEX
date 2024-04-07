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
            'path': './dataset/',
            'dataset_file': pd.DataFrame(
                data=[

                    ['raw-all-human-Margarida.h5ad', True, 'gene_matches_human2macaque.csv',
                     'raw-all-macaque-Margarida.h5ad', False],

                    ['raw-all-human-Margarida.h5ad', True, 'gene_matches_human2mouse.csv',
                     'raw-all-mouse-Margarida.h5ad', False],

                    ['raw-all-human-Margarida.h5ad', True, 'gene_matches_human2rat.csv',
                     'raw-all-rat-Margarida.h5ad', False],

                    ['raw-all-human-Margarida.h5ad', True, 'gene_matches_human2rabbit.csv',
                     'raw-all-rabbit-Margarida.h5ad', False],

                    ['raw-all-human-Margarida.h5ad', True, 'gene_matches_human2opossum.csv',
                     'raw-all-opossum-Margarida.h5ad', False],

                    ['raw-all-human-Margarida.h5ad', True, 'gene_matches_human2chicken.csv',
                     'raw-all-chicken-Margarida.h5ad', False],

                    # ['raw-all-human-Margarida.h5ad', True, 'gene_matches_human2human.csv',
                    #  'raw-retina-human-lubulk.h5ad', False],
                    #
                    # ['raw-all-human-Margarida.h5ad', True, 'gene_matches_human2mouse2.csv',
                    #  'raw-retina-mouse-clarkbulk.h5ad', False],

                ],
                columns=['source', 'source label', 'relationship', 'destination', 'destination label']),
            'graph_mode': 'undirected',
            # 'feature_gene': 'all',  # can impute all_1v1_gene
            'feature_gene': 'HIG',  # can impute HIG
            'sample_ratio': 1,  # default 1, set to ratio of (0, 1] to down sample the dataset
            'get_balance': 'False'  # set ref and query with the same cell type
        },

        'train': {
            'device': 'cuda:0',  # cpu or cuda
            'train_mode': 'full_batch',
            # 'train_mode': 'mini_batch',
            'dim_hidden': 128,
            'batch_size': 1024,
            'gnn_layer_num': 2,
            'epoch_pretrain': 200,
            'epoch_train': 200,
            'encoder': 'GCN',
            # 'encoder': 'GAT',
            'decoder': 'Dot',
            'classifier': 'GAT',
            'res': True,
            'share': True,
            'cluster': False,
            'epoch_cluster': 10,
            'cluster_num': 5,

            'domain': False,
            'reconstruct': False,

        },

        'postprocess': {}
    }
