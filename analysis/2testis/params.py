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

                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2chimpanzee.csv',
                     'raw-testis-chimpanzee-Murat.h5ad', False],
                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2bonobo.csv',
                     'raw-testis-bonobo-Murat.h5ad', False],
                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2gorilla.csv',
                     'raw-testis-gorilla-Murat.h5ad', False],
                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2gibbon.csv',
                     'raw-testis-gibbon-Murat.h5ad', False],
                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2macaque.csv',
                     'raw-testis-macaque-Murat.h5ad', False],
                    
                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2marmoset.csv',
                     'raw-testis-marmoset-Murat.h5ad', False],
                    
                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2mouse.csv',
                     'raw-testis-mouse-Murat.h5ad', False],

                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2opossum.csv',
                     'raw-testis-opossum-Murat.h5ad', False],
                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2platypus.csv',
                     'raw-testis-platypus-Murat.h5ad', False],
                    ['raw-testis-human-Murat.h5ad', True, 'gene_matches_human2chicken.csv',
                     'raw-testis-chicken-Murat.h5ad', False],

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
            'dim_hidden': 128,
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
