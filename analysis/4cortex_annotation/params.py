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
                    ['raw-brain-human-Lake.h5ad', True, 'gene_matches_human2mouse.csv',
                     'raw-brain-mouse-Chen.h5ad', False],
                    ['raw-brain-human-Lake.h5ad', True, 'gene_matches_human2lizard.csv',
                     'raw-brain-lizard-Tosches.h5ad', False],
                    ['raw-brain-human-Lake.h5ad', True, 'gene_matches_human2turtle.csv',
                     'raw-brain-turtle-Tosches.h5ad', False],


                    # ['raw-brain-human-Lake.h5ad', True, 'gene_matches_human2mouse1.csv',
                    #  'raw-brain-mouse-campbell.h5ad', False],
                    # ['raw-brain-human-Lake.h5ad', True, 'gene_matches_human2mouse2.csv',
                    #  'raw-brain-mouse-myeloid.h5ad', False],
                    #
                    # ['raw-brain-human-Lake.h5ad', True, 'gene_matches_human2mouse3.csv',
                    #  'raw-brain-mouse-tasic18.h5ad', False],

                    # 全部连接
                    # ['raw-brain-mouse-Chen.h5ad', False, 'gene_matches_mouse2turtle.csv',
                    #  'raw-brain-turtle-Tosches.h5ad', False],
                    # ['raw-brain-mouse-Chen.h5ad', False, 'gene_matches_mouse2lizard.csv',
                    #  'raw-brain-lizard-Tosches.h5ad', False],
                    # ['raw-brain-turtle-Tosches.h5ad', False, 'gene_matches_turtle2lizard.csv',
                    #  'raw-brain-lizard-Tosches.h5ad', False],
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
            # 'train_mode': 'full_batch',
            'train_mode': 'mini_batch',
            'dim_hidden': 128,
            'batch_size': 1024,
            'gnn_layer_num': 2,
            'epoch_integration': 10,
            'epoch_annotation': 10,
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

