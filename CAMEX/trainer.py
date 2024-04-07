# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/7/24 19:52
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : trainer.py
# @IDE     : PyCharm

import pandas as pd
import copy

import torch
import scanpy as sc

from CAMEX.train_untils import *
from CAMEX.model import *


class Trainer(object):
    def __init__(self, adata_whole, dgl_data, **kwargs):
        # params
        self.params = kwargs
        assert self.params['train_mode'] in {'full_batch', 'mini_batch'}
        self.train_mode = self.params['train_mode']
        self.use_domain = self.params['domain']

        # whole_adata
        self.adata_whole = adata_whole
        # dgl
        self.dgl_data = dgl_data
        self.cluster_centers = {}
        self.cluster_num = self.params['cluster_num']
        self.cluster_epoch = self.params['epoch_cluster']

        # model
        self.encoder = None
        self.decoder_dot = None
        self.classifier = None
        self.domain = None
        self.model = None
        self._generate_model()

        # optimizer
        self.optimizer_dot = None
        self.optimizer_gat = None
        self.optimizer_class = None
        self.optimizer_domain = None
        self.scheduler = None
        self._generate_optimizer()

        # loss_func
        self.loss_reconstruct_graph = None
        self.loss_reconstruct_feature = None
        self.loss_classification = None
        self.loss_domain = None
        self._generate_loss_func()

        # log
        self.epoch_current_pretrain = 0
        self.log_pretrain = {'best_epoch': 0,  'best_hidden': {},  'evaluation': pd.DataFrame(),
                             'best_model_params': []}
        self.epoch_current_train = 0
        self.log_train = {'best_train_acc': 0, 'best_test_acc': 0, 'best_epoch': 0,  'best_class': {},
                          'best_hidden': {}, 'evaluation': pd.DataFrame(), 'best_model_params': [], 'best_test_AMI': 0}

    def _generate_model(self):
        # encoder
        if self.params['encoder'] == 'GCN':
            self.encoder = RGNNEncoder(self.dgl_data.graph.canonical_etypes,
                                       dim_in=self.dgl_data.feature_dim,
                                       dim_hidden=self.params['dim_hidden'],
                                       layer_num=self.params['gnn_layer_num'],
                                       res=self.params['res'],
                                       # graph=self.dgl_data.graph
                                       )
        elif self.params['encoder'] == 'GAT':
            self.encoder = RGATEncoder(self.dgl_data.graph.canonical_etypes,
                                       dim_in=self.dgl_data.feature_dim,
                                       dim_hidden=self.params['dim_hidden'],
                                       layer_num=self.params['gnn_layer_num'],
                                       )
        else:
            print('not implementedError yet')
            raise NotImplementedError

        # decoder
        self.decoder_dot = DotDecoder()

        # classifier
        if self.params['classifier'] == 'GAT':
            self.classifier = GATClassifier(self.dgl_data.graph.canonical_etypes,
                                            dim_hidden=self.params['dim_hidden'],
                                            dim_out=self.dgl_data.class_num,
                                            )
        else:
            print('not implementedError yet')
            raise NotImplementedError

        self.domain = MLPClassifier(dim_hidden=self.params['dim_hidden'], dim_out=len(self.dgl_data.data_type))
        self.cluster_num_dict = {name: self.cluster_num for name, data in self.dgl_data.graph.ndata['cluster_label'].items()}
        self.cluster_classifier = nn.ModuleDict({name: ClusterClassifier(dim_hidden=self.params['dim_hidden'], dim_out=data)
                                     for name, data in self.cluster_num_dict.items()})
        self.mlp = nn.ModuleDict({name: MLPDecoder(dim_hidden=self.params['dim_hidden'], dim_out=self.dgl_data.feature_dim)
                                  for name, data in self.dgl_data.graph.ndata['feature'].items()})
        # model
        self.model = GAE(self.encoder, self.decoder_dot, self.classifier, self.domain, self.cluster_classifier, self.mlp)
        self.model.reset_parameters()

    def _generate_optimizer(self):
        # dot无参数，只优化encoder
        self.optimizer_model = torch.optim.Adam(self.model.parameters(), weight_decay=0.001)
        self.optimizer_domain = torch.optim.Adam(self.domain.parameters(), weight_decay=0.001)
        self.optimizer_generator = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': 1e-3},
                                                     {'params': self.decoder_dot.parameters(), 'lr': 1e-3},
                                                     {'params': self.mlp.parameters(), 'lr': 1e-3}], weight_decay=0.001)
        self.optimizer_classifier = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': 1e-3},
                                                      {'params': self.classifier.parameters(), 'lr': 1e-3}], weight_decay=0.001)
        self.optimizer_cluster = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': 1e-3},
                                                   {'params': self.cluster_classifier.parameters(), 'lr': 1e-3}], weight_decay=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_model, step_size=5, gamma=0.5)

    def _generate_loss_func(self):
        self.loss_reconstruct_graph = torch.nn.BCEWithLogitsLoss()
        self.loss_reconstruct_feature = sce_loss
        self.loss_classification = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_domain = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_cluster = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_mlp = torch.nn.MSELoss()

    def _batch_pretrain(self, graph, device, step='d'):
        # 释放显存
        # to device
        self.model.to(device)
        # self.model.to_device(device)
        graph = graph.to(device)

        loss_total = 0

        if self.params['domain']:
            # encode
            h_hidden = self.encoder(graph, graph.ndata['feature'])
            # domain
            prediction = self.domain(h_hidden, 1)
            # domain label TODO
            domain_label = {}
            counter = 0
            for name, data in prediction.items():
                if name.endswith('cell'):
                    label = torch.tensor([counter] * len(data)).long().to(device)
                    domain_label[name] = label
                    counter += 1
            # loss
            loss_domain = [self.loss_domain(data, domain_label[name]) for name, data in prediction.items() if
                           name.endswith('cell')]
            loss_domain = sum(loss_domain) * 0.1

            if step == 'd':
                self.optimizer_domain.zero_grad()
                # backward
                loss_domain.backward()
                torch.nn.utils.clip_grad_value_(self.domain.parameters(), clip_value=1)    # TODO
                torch.nn.utils.clip_grad_norm_(self.domain.parameters(), max_norm=20, norm_type=2)
                self.optimizer_domain.step()
                return loss_domain
        else:
            loss_domain = 0

        # forward
        # to device
        self.decoder_dot.device = device
        # encode
        h_hidden = self.encoder(graph, graph.ndata['feature'])
        # decode
        out_positive, out_negative = self.decoder_dot(graph, h_hidden)

        # reconstruct loss
        label_p = {name: torch.ones(len(data)).to(device) for name, data in out_positive.items() if data.size()}
        label_n = {name: torch.zeros(len(data)).to(device) for name, data in out_negative.items() if data.size()}

        loss_reconstruct_p = [torch.nn.BCEWithLogitsLoss()(predict_p, label_p[edge])
                              for edge, predict_p in out_positive.items() if predict_p.size()]
        loss_reconstruct_n = [torch.nn.BCEWithLogitsLoss()(predict_n, label_n[edge])
                              for edge, predict_n in out_negative.items() if predict_n.size()]
        loss_reconstruct_graph = sum([item for item in loss_reconstruct_p if item > 0]) + \
                                 sum([item for item in loss_reconstruct_n if item > 0])

        if self.params['reconstruct']:
            h_hat = {name: self.mlp[name](data) for name, data in h_hidden.items() if name.endswith('cell')}
            loss_mlp = [self.loss_mlp(data, graph.ndata['feature'][name]) for name, data in h_hat.items()]
            loss_mlp = sum(loss_mlp)
        else:
            loss_mlp = 0
        loss = loss_reconstruct_graph - loss_domain + loss_mlp

        # backward
        self.optimizer_generator.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        self.optimizer_generator.step()

        loss_total += loss.item()

        # if self.params['cluster'] and self.epoch_current_pretrain > self.cluster_epoch:
        #     # encode
        #     h_hidden = self.encoder(graph, graph.ndata['feature'])
        #     # prediction
        #     prediction = {name: self.cluster_classifier[name](data, 1) for name, data in h_hidden.items() if name.endswith('cell')}
        #     # cluster_label
        #     cluster_label = graph.ndata['cluster_label']
        #     # loss
        #     loss_cluster = [self.loss_cluster(data, cluster_label[name]) for name, data in prediction.items()
        #                     if name.endswith('cell')]
        #     loss_cluster = sum(loss_cluster) * 0.1
        #     self.optimizer_cluster.zero_grad()
        #     # backward
        #     loss_cluster.backward()
        #     torch.nn.utils.clip_grad_value_(self.encoder.parameters(), clip_value=1)
        #     torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=20, norm_type=2)
        #     self.optimizer_cluster.step()

        # if self.params['cluster'] and self.epoch_current_pretrain >= 10:
        #     # 设置类中心
        #     if self.epoch_current_pretrain == 10:
        #         from torch.nn.parameter import Parameter
        #         self.model.cluster_classifier.params_dict = nn.ParameterDict({name: Parameter(torch.Tensor(centers).to(device))
        #                                                                       for name, centers in self.cluster_centers.items()})
        #     # encode
        #     h_hidden = self.encoder(graph, graph.ndata['feature'])
        #     # prediction
        #     q, p = self.cluster(h_hidden)
        #     loss_cluster = [F.kl_div(data.log(), p[name], reduction='batchmean') for name, data in q.items()]
        #     # loss
        #     loss_cluster = sum(loss_cluster) * 0.1
        #     self.optimizer_cluster.zero_grad()
        #     # backward
        #     loss_cluster.backward()
        #     torch.nn.utils.clip_grad_value_(self.encoder.parameters(), clip_value=1)
        #     torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=20, norm_type=2)
        #     self.optimizer_cluster.step()

        return loss_total

    def _batch_train(self, graph, device, step='d'):
        # 释放显存
        # torch.cuda.empty_cache()
        # to device
        self.model.to(device)
        graph = graph.to(device)

        # if self.params['domain']:
        #     # encode
        #     h_hidden = self.encoder(graph, graph.ndata['feature'])
        #     # domain
        #     prediction = self.domain(h_hidden, 1)
        #     # domain label TODO
        #     domain_label = {}
        #     counter = 0
        #     for name, data in prediction.items():
        #         if name.endswith('cell'):
        #             label = torch.tensor([counter] * len(data)).long().to(device)
        #             domain_label[name] = label
        #             counter += 1
        #     # loss
        #     loss_domain = [self.loss_domain(data, domain_label[name]) for name, data in prediction.items() if
        #                    name.endswith('cell')]
        #     loss_domain = sum(loss_domain) * 0.1
        #
        #     if step == 'd':
        #         self.optimizer_domain.zero_grad()
        #         # backward
        #         loss_domain.backward()
        #         torch.nn.utils.clip_grad_value_(self.domain.parameters(), clip_value=1)    # TODO
        #         torch.nn.utils.clip_grad_norm_(self.domain.parameters(), max_norm=20, norm_type=2)
        #         self.optimizer_domain.step()
        #         return loss_domain
        # else:
        #     loss_domain = 0

        # zero_grad
        # self.optimizer_classifier.zero_grad()
        self.optimizer_classifier.zero_grad()
        # 数据集类型，标记哪些训练
        data_ref = [name for name, ty in self.dgl_data.data_type.items() if ty == 'reference']

        # encode
        h_hidden = self.encoder(graph, graph.ndata['feature'])
        # class
        h_class = self.classifier(graph, h_hidden)

        # loss list
        loss_class_list = [self.loss_classification(h_class[name + 'cell'], graph.ndata['cell_label_cl'][name + 'cell'])
                           for name in data_ref]
        loss_class = sum(loss_class_list)
        loss = loss_class
        # loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1)  # 裁剪梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        # step
        # self.optimizer_classifier.step()
        self.optimizer_classifier.step()
        # loss
        loss = loss.item()
        return loss

    def pretrain(self):
        self._generate_model()
        self._generate_optimizer()
        print('--------------------------------------------- pretrain ---------------------------------------------')
        device = self.params['device']
        n_epoch = self.params['epoch_pretrain']

        for epoch in range(n_epoch):
            # 当前epoch
            self.epoch_current_pretrain = epoch
            # train
            self.model.train()
            loss_sum = 0

            # if epoch >= 5:
            #     # TODO 软聚类
            #     self.clustering()

            # full_batch，整图回传优化一次
            if self.train_mode == 'full_batch':
                # batch == epoch
                if epoch >= 20 and self.params['domain']:
                    loss = self._batch_pretrain(self.dgl_data.graph, device, step='d')
                loss = self._batch_pretrain(self.dgl_data.graph, device, step='g')
                loss_sum = loss

            # mini_batch，每个批次回传优化
            elif self.train_mode == 'mini_batch':
                # sampler
                sampler_list = [3]  # TODO 超参数
                # sampler_list = [5] * (1 + self.params['gnn_layer_num'] + 1)     # 每层只取5个邻居
                sampler = dgl.dataloading.ShaDowKHopSampler(sampler_list)
                # train_nids是全部的，loss时做选择
                train_nids = self.dgl_data.graph.ndata['index']
                # dataloader
                dataloader = dgl.dataloading.NodeDataLoader(self.dgl_data.graph,
                                                            train_nids,
                                                            sampler,
                                                            batch_size=self.params['batch_size'],
                                                            shuffle=True,
                                                            drop_last=True)
                for batch_num, (input_nodes, output_nodes, block) in enumerate(dataloader):
                    # domain alpha
                    p = float(batch_num + epoch * len(dataloader)) / n_epoch / len(dataloader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    # train
                    if epoch >= 5 and self.params['domain']:
                        loss = self._batch_pretrain(block, device, step='d')
                    loss = self._batch_pretrain(block, device, step='g')
                    loss_sum += loss
                loss_sum = loss_sum / len(dataloader)

            print(f'epoch: {self.epoch_current_pretrain}, loss: {loss_sum}')

            # eval
            self._evaluation_pretrain(loss_sum)
            self.adata_whole.obsm['cell_pretrain_hidden'] = np.concatenate(
                [self.log_pretrain['best_hidden'][name + 'cell'] for name in self.adata_whole.uns['data_order']])

        # save
        self._store_pretrain()

    def train(self):
        self._generate_model()
        self._generate_optimizer()
        print('--------------------------------------------- train ---------------------------------------------')
        # train
        device = self.params['device']
        n_epoch = self.params['epoch_train']

        for epoch in range(n_epoch):
            # # TODO test
            # self.get_feature_importance()
            # model先放入cuda，data可能一次放不下，分full_batch和mini_batch
            self.model.to(device)
            # 当前epoch
            self.epoch_current_train = epoch
            # train
            self.model.train()
            # loss_sum
            loss_sum = 0

            # full_batch，整图回传优化一次
            if self.train_mode == 'full_batch':
                loss = self._batch_train(self.dgl_data.graph, device, step='g')
                loss_sum += loss

            # mini_batch，每个批次回传优化
            elif self.train_mode == 'mini_batch':
                # sampler
                sampler_list = [3]  # TODO 超参数，和邻居的距离有关
                # sampler_list = [5] * (1 + self.params['gnn_layer_num'] + 1)     # 每层只取5个邻居
                sampler = dgl.dataloading.ShaDowKHopSampler(sampler_list)
                # train_nids是全部的，loss时做选择
                data_ref = [name for name, ty in self.dgl_data.data_type.items() if ty == 'reference']
                train_nids = self.dgl_data.graph.ndata['index']
                # train_nids = {item + 'cell': train_nids[item + 'cell'] for item in data_ref}    # TODO 节约时间
                if len(train_nids) == 0:
                    print('the num of ref is 0!')
                    raise NotImplementedError
                # dataloader
                dataloader = dgl.dataloading.NodeDataLoader(self.dgl_data.graph,
                                                            train_nids,
                                                            sampler,
                                                            batch_size=self.params['batch_size'],
                                                            shuffle=True,
                                                            drop_last=True)
                loss_sum = 0
                for batch_num, (input_nodes, output_nodes, block) in enumerate(dataloader):
                    loss = self._batch_train(block, device)
                    loss_sum += loss

                loss_sum = loss_sum

            # eval
            print(f'epoch: {self.epoch_current_train}, loss: {loss_sum}')
            self._evaluation_train()

            # checkpoint
            if self.epoch_current_train - self.log_train['best_epoch'] > int(n_epoch / 10):
            # if self.epoch_current_train - self.log_train['best_epoch'] > 5:
                self.model.load_state_dict(self.log_train['best_model_params'])  # load model params
                self.log_train['best_epoch'] = self.epoch_current_train  # 当前epoch

        # save
        self._store_train()

    def _evaluation_pretrain(self, loss_sum):
        """
        评估时不需要model.eval
        :return:
        """
        # dataloader or one batch
        with torch.no_grad():
            model = self.model.to('cpu')
            graph_temp = self.dgl_data.graph.to('cpu')
            node_feature_temp = graph_temp.ndata['feature']
            hidden = model.encoder(graph_temp, node_feature_temp)
            del graph_temp

        # if self.epoch_current_pretrain == 0 and self.epoch_current_pretrain % 2 == 0:   # TODO
        if self.params['cluster'] and self.epoch_current_pretrain == self.cluster_epoch:   # TODO
            print('clustering...')
            self.clustering(hidden)

        hidden = tensor_to_numpy(hidden)

        # loss
        evaluation_dict = {'loss': loss_sum}
        evaluation = pd.DataFrame(evaluation_dict, index=[self.epoch_current_pretrain])
        self.log_pretrain['evaluation'] = pd.concat([self.log_pretrain['evaluation'], evaluation])

        # save
        self.log_pretrain['best_hidden'] = hidden
        self.log_pretrain['best_model_params'] = self.model.state_dict()

    def _evaluation_train(self):
        """
        无需model.eval
        :return:
        """
        # eval prepare, forward
        with torch.no_grad():
            self.model.train()
            self.model.to('cpu')
            graph_temp = self.dgl_data.graph.to('cpu')
            h_hidden = self.encoder(graph_temp, graph_temp.ndata['feature'])
            h_class = self.classifier(graph_temp, h_hidden)
            del graph_temp

        with torch.no_grad():
            self.model.eval()
            self.model.to('cpu')
            graph_temp = self.dgl_data.graph.to('cpu')
            h_hidden_eval = self.encoder(graph_temp, graph_temp.ndata['feature'])
            h_class_eval = self.classifier(graph_temp, h_hidden_eval)
            del graph_temp

        # label true
        y_true = self.dgl_data.graph.ndata['cell_label_class']
        data_ref = [name for name, ty in self.dgl_data.data_type.items() if ty == 'reference']
        data_query = [name for name, ty in self.dgl_data.data_type.items() if ty == 'query']
        y_true_train = tensor_to_numpy({name: label for name, label in y_true.items() if name[: -4] in data_ref})
        y_true_test = tensor_to_numpy({name: label for name, label in y_true.items() if name[: -4] in data_query})
        y_cluster = tensor_to_numpy(self.dgl_data.graph.ndata['cluster_label'])     # TODO

        # label predict
        y_predict_prob = {name: nn.Softmax(dim=-1)(data) for name, data in h_class.items()}
        y_predict = tensor_to_numpy({name: torch.max(data, -1)[1] for name, data in y_predict_prob.items() if name.endswith('cell')})     # TODO
        y_predict_prob_train = {name: y_predict_prob[name] for name, label in y_true_train.items()}
        y_predict_prob_test = {name: y_predict_prob[name] for name, label in y_true_test.items()}

        # 不设置阈值
        y_predict_train = tensor_to_numpy({dataset_name: torch.max(label, -1)[1] for dataset_name, label in
                                           y_predict_prob_train.items()})
        y_predict_test = tensor_to_numpy({dataset_name: torch.max(label, -1)[1] for dataset_name, label in
                                          y_predict_prob_test.items()})
        # # 设置阈值，只有概率大于0.5才确定为预测值，否则设置为unknown0
        # unknown_num = self.adata_whole.uns['cell_type']['unknown']
        # y_predict_train = cal_pred_with_unknown(y_predict_prob_train, unknown_num)
        # y_predict_test = cal_pred_with_unknown(y_predict_prob_test, unknown_num)

        # metrics
        train_acc = get_acc(y_true_train, y_predict_train)  # dict
        train_acc_list = [value for value in train_acc.values()]
        train_f1_weighted = get_f1(y_true_train, y_predict_train, average='weighted')

        test_acc = get_acc(y_true_test, y_predict_test)
        test_acc_list = [value for value in test_acc.values()]
        test_f1_weighted = get_f1(y_true_test, y_predict_test, average='weighted')

        # AMI
        test_AMI = get_ami(y_cluster, y_predict)     # TODO 训练集的ami 或者 聚类标签和预测标签的ami

        # if round(sum(train_acc_list), 4) > self.log_train['best_train_acc'] or self.epoch_current_train == 0:  # 第一次进入时
        if sum(test_AMI.values()) > self.log_train['best_test_AMI'] or self.epoch_current_train == 0:  # 第一次进入时
            self.log_train['best_train_acc'] = round(sum(train_acc_list), 4)
            self.log_train['best_test_acc'] = copy.deepcopy(round(sum(test_acc_list), 4))
            self.log_train['best_epoch'] = copy.deepcopy(self.epoch_current_train)
            self.log_train['best_class'] = tensor_to_numpy(h_class)  # class作为embedding
            self.log_train['best_hidden'] = tensor_to_numpy(h_hidden)  # h_hidden作为embedding
            self.log_train['best_hidden_eval'] = tensor_to_numpy(h_hidden_eval)  # h_hidden作为embedding
            self.log_train['best_model_params'] = self.model.state_dict()  # save model params
            self.log_train['best_test_AMI'] = sum(test_AMI.values())  #

        # save evaluation
        evaluation_dict = {}
        evaluation_dict.update(train_acc)
        evaluation_dict.update(train_f1_weighted)
        evaluation_dict.update(test_acc)
        evaluation_dict.update(test_f1_weighted)
        evaluation_dict.update(test_f1_weighted)
        evaluation_dict.update(test_AMI)

        evaluation = pd.DataFrame(evaluation_dict, index=[self.epoch_current_train])
        self.log_train['evaluation'] = pd.concat([self.log_train['evaluation'], evaluation])

        # print
        info = f'train_acc: {train_acc}, test_acc: {test_acc}, train_ami:{test_AMI}, best_epoch: {self.log_train["best_epoch"]}'
        print(info)
        return

    def _store_pretrain(self):
        # cell embedding
        cell_h_hidden = []
        for k in self.adata_whole.uns['data_order']:
            cell_h_hidden.extend(self.log_pretrain['best_hidden'][str(k) + 'cell'])
        self.adata_whole.obsm['cell_pretrain_hidden'] = np.array(cell_h_hidden)

        # gene embedding，adata中不能保存，单独存放为pt文件
        gene_embedding_dict = {}
        for name, gene_name in self.dgl_data.node_gene.items():
            gene_embedding = self.log_pretrain['best_hidden'][name + 'gene']
            gene_embedding_dict[name] = pd.DataFrame(data=gene_embedding, index=gene_name)
        import scanpy as sc
        embedding = pd.concat([v for k, v in gene_embedding_dict.items()])
        gene_pretrain = sc.AnnData(X=embedding.to_numpy(), obs=pd.DataFrame(index=embedding.index))
        gene_pretrain.obs.loc[:, 'batch'] = pd.concat([pd.DataFrame(len(v) * [k]) for k, v in gene_embedding_dict.items()], ignore_index=True).to_numpy()
        gene_pretrain.obsm['X_emb'] = embedding.to_numpy()
        gene_pretrain.write_h5ad(self.params['log_path'] + 'gene_pretrain_hidden.h5ad')
        # torch.save(gene_embedding_dict, self.params['log_path'] + 'gene_embedding_pretrain.pt')

        # save evaluation
        self.log_pretrain['evaluation'].to_csv(f'{self.params["log_path"]}/evaluation_pretrain.csv', index=False)
        # save model_params
        torch.save(self.log_pretrain['best_model_params'], f'{self.params["log_path"]}/model_params_pretrain.pt')

    def _store_train(self):
        # save cell embedding
        cell_class = []
        cell_hidden = []
        cell_hidden_eval = []
        for k in self.adata_whole.uns['data_order']:
            cell_class.extend(self.log_train['best_class'][str(k) + 'cell'])
            cell_hidden.extend(self.log_train['best_hidden'][str(k) + 'cell'])
            cell_hidden_eval.extend(self.log_train['best_hidden_eval'][str(k) + 'cell'])
        # 存到obsm中
        self.adata_whole.obsm['cell_train_class'] = np.array(cell_class)
        self.adata_whole.obsm['cell_train_hidden'] = np.array(cell_hidden)
        self.adata_whole.obsm['cell_train_hidden_eval'] = np.array(cell_hidden_eval)

        # # 存到obs
        # prob = nn.Softmax()(torch.tensor(self.adata_whole.obsm['cell_train_class'])).numpy()
        # prob_df = pd.DataFrame(data=prob, columns=self.adata_whole.uns['cell_type'].values())

        # save gene embedding
        gene_embedding_hidden_dict = {}
        gene_embedding_class_dict = {}
        for name, gene_name in self.dgl_data.node_gene.items():
            gene_embedding_hidden = self.log_train['best_hidden'][name + 'gene']
            gene_embedding_hidden_dict[name] = pd.DataFrame(data=gene_embedding_hidden, index=gene_name)

            gene_embedding_class = self.log_train['best_class'][name + 'gene']
            gene_embedding_class_dict[name] = pd.DataFrame(data=gene_embedding_class, index=gene_name)

        import scanpy as sc
        embedding = pd.concat([v for k, v in gene_embedding_hidden_dict.items()])
        gene_train_hidden = sc.AnnData(X=embedding.to_numpy(), obs=pd.DataFrame(index=embedding.index))
        gene_train_hidden.obs.loc[:, 'batch'] = pd.concat(
            [pd.DataFrame(len(v) * [k]) for k, v in gene_embedding_hidden_dict.items()], ignore_index=True).to_numpy()
        gene_train_hidden.obsm['X_emb'] = embedding.to_numpy()
        gene_train_hidden.write_h5ad(self.params['log_path'] + 'gene_train_hidden.h5ad')
        # torch.save(gene_embedding_hidden_dict, self.params['log_path'] + 'gene_embedding_train_hidden.pt')
        embedding = pd.concat([v for k, v in gene_embedding_class_dict.items()])
        gene_train_class = sc.AnnData(X=embedding.to_numpy(), obs=pd.DataFrame(index=embedding.index))
        gene_train_class.obs.loc[:, 'batch'] = pd.concat(
            [pd.DataFrame(len(v) * [k]) for k, v in gene_embedding_class_dict.items()], ignore_index=True).to_numpy()
        gene_train_class.obsm['X_emb'] = embedding.to_numpy()
        gene_train_class.write_h5ad(self.params['log_path'] + 'gene_train_class.h5ad')
        # torch.save(gene_embedding_class_dict, self.params['log_path'] + 'gene_embedding_train_class.pt')

        # save evaluation
        self.log_train['evaluation'].to_csv(f'{self.params["log_path"]}/evaluation_train.csv', index=False)
        # save model_params
        torch.save(self.log_train['best_model_params'], f'{self.params["log_path"]}/model_params_train.pt')

    def predict(self, device='cpu'):
        """
        数据集和模型相关，所以不能解耦
        在device cpu or gpu 预测全部的数据
        返回值为不同层的embedding
        :return:
        """
        # eval prepare, forward
        self.model.to(device)
        graph_temp = self.dgl_data.graph.to('cpu')
        h_hidden = self.encoder(graph_temp, graph_temp.ndata['feature'])
        h_class = self.classifier(graph_temp, h_hidden)
        del graph_temp
        return h_hidden, h_class

    def get_feature_importance(self, cell_type: int = 0, device='cpu'):
        """
        输入指定类型的细胞，如果不输入则是预测类别
        返回对应的top k gene
        :param device:
        :param cell_type:
        :return:
        """
        # # TODO method1
        # # 获得抽取器
        # from torchcam.methods import GradCAM, GradCAMpp, ISCAM, LayerCAM, SSCAM, ScoreCAM, SmoothGradCAMpp, XGradCAM
        # cam_extractor = GradCAM(self.model, 'encoder')
        #
        # # 获得预测结果，或者自己产生
        # self.model.to(device)
        # graph_temp = self.dgl_data.graph.to(device)
        # h_class = self.model(graph_temp, graph_temp.ndata['feature'])
        # y_predict = tensor_to_numpy({dataset_name: torch.max(label, -1)[1] for dataset_name, label in h_class.items()})
        #
        # # 看看能不能接受字典，不行就生成表达式
        # activation_map_cam = cam_extractor(y_predict, h_class)  # TODO
        # del activation_map_cam

        # # TODO method2
        # from pytorch_grad_cam import GradCAM
        # target_layers = [self.model.encoder]
        # cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=False)
        # from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        # targets = [ClassifierOutputTarget(22)]
        # # 获得预测结果，或者自己产生
        # graph_temp = self.dgl_data.graph.to(device)
        # cam_map = cam(input_tensor=(graph_temp, graph_temp.ndata['feature']), targets=targets)[0]  # 不加平滑

    def clustering(self, feature):
        from sklearn.cluster import KMeans
        for name, data in feature.items():
            if name.endswith('cell'):  # TODO gene and cell
                kmeans = KMeans(n_clusters=self.cluster_num, n_init=20)
                y_pred = kmeans.fit_predict(data.cpu().numpy())
                self.cluster_centers[name] = kmeans.cluster_centers_
                self.dgl_data.graph.nodes[name].data['cluster_label'] = torch.tensor(y_pred, device='cpu', dtype=torch.int64)


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def add_noise(graph):
    # add noise 1, drop and add edge, useless
    from dgl import DropEdge, AddEdge, FeatMask
    transform_d = DropEdge(p=0.2)
    transform_a = AddEdge(ratio=0.2)
    transform_f = FeatMask(p=0.2)
    graph = transform_d(graph)
    graph = transform_a(graph)
    graph = transform_f(graph)
    return graph
