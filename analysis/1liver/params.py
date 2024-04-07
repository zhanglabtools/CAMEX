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
                    # all
                    ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2monkey.csv',
                     'raw-liver-monkey-Martin.h5ad', False],

                    ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2mouse.csv',
                     'raw-liver-mouse-Martin.h5ad', False],

                    ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2zebrafish.csv',
                     'raw-liver-zebrafish-ggj5.h5ad', False],

                    # ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2chicken.csv',
                    #  'raw-liver-chicken-Martin.h5ad', False],
                    #
                    # ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2pig.csv',
                    #  'raw-liver-pig-Martin.h5ad', False],

                    # ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2hamster.csv',
                    #  'raw-liver-hamster-Martin.h5ad', False],
                    #
                    # ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2zebrafish2.csv',
                    #  'raw-liver-zebrafish-Martin.h5ad', False],

                    # ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2cat.csv',
                    #  'raw-liver-cat-ggj.h5ad', False],
                    #
                    # ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2tiger.csv',
                    #  'raw-liver-tiger-ggj.h5ad', False],

                    # ['raw-liver-human-Martin.h5ad', True, 'gene_matches_human2pig.csv',
                    #  'raw-liver-pig-wang.h5ad', False],

                    # # immune
                    # ['raw-liver-chicken-Martin.h5ad', True, 'gene_matches_chicken2hamster.csv',
                    #  'raw-liver-hamster-Martin.h5ad', False],
                    #
                    # ['raw-liver-chicken-Martin.h5ad', True, 'gene_matches_chicken2pig.csv',
                    #  'raw-liver-pig-Martin.h5ad', False],
                    #
                    # ['raw-liver-chicken-Martin.h5ad', True, 'gene_matches_chicken2zebrafish.csv',
                    #  'raw-liver-zebrafish-Martin.h5ad', False],

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
            'train_mode': 'mini_batch',
            'dim_hidden': 16,
            'batch_size': 1024,
            'gnn_layer_num': 2,
            'epoch_pretrain': 10,
            'epoch_train': 10,
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
            'reconstruct': True,

        },

        'postprocess': {}
    }

