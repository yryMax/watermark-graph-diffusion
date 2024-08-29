import os
import pathlib
import pickle
import random
from typing import Optional

import Graph_Sampling
import networkx as nx
import numpy as np
import torch
import torch_geometric.data
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from src.datasets.abstract_dataset import (AbstractDataModule,
                                           AbstractDatasetInfos)
from src.g2gcompress import Graph2GraphPairCompress


class SocialGraphDataset(InMemoryDataset):
    url = 'https://drive.usercontent.google.com/download?id={}&confirm=t'
    dataset_list = {
        'facebook': '1zWnz32-0fWF8_xPLqzIDoL4XYphR_bNl',
        'flickr': '1xTWlVH5uchKuYOkv0pPddEYJPg6ZC5lA'
    }
    file_idx = {'train': 0, 'val': 1, 'test': 2}

    def __init__(self, dataset_name, split, root,
                 num_graphs: int, min_nodes: int, max_nodes: int,
                 g2gc: Optional[Graph2GraphPairCompress], seed=1234,
                 transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.seed = seed
        self.g2gc = g2gc
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(
            self.processed_paths[self.file_idx[self.split]])

    @property
    def raw_file_names(self):
        return ['processed_graph.pkl']

    @property
    def processed_file_names(self):
        return [
            (   f'{split}-{self.num_graphs}g-{self.min_nodes}_{self.max_nodes}n'
                f'-{self.seed}s{"-g2gc" if self.g2gc else ""}.pt')
            for split in self.file_idx
        ]

    def download(self):
        if self.dataset_name not in self.dataset_list:
            raise ValueError(f'Unknown dataset {self.dataset_name}')

        url = self.url.format(self.dataset_list[self.dataset_name])
        download_url(url, self.raw_dir, filename=self.raw_paths[0])

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            g_nx = pickle.load(f)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        np_rng = np.random.default_rng(self.seed)
        random.seed(self.seed)
        obj = Graph_Sampling.SRW_RWF_ISRW()
        data_list = []

        for n in np_rng.integers(low=self.min_nodes, high=self.max_nodes, size=self.num_graphs):
            sample_nx = obj.random_walk_induced_graph_sampling(g_nx, n)
            adj = nx.adjacency_matrix(sample_nx)
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.from_scipy_sparse_matrix(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)

            if self.g2gc:
                data = self.g2gc.encode(data)
                data.n_nodes[0] = data.x.size(0)

            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

        torch.save(self.collate(data_list[:train_len]),
                   self.processed_paths[self.file_idx['train']])
        torch.save(self.collate(data_list[train_len:train_len + val_len]),
                   self.processed_paths[self.file_idx['val']])
        torch.save(self.collate(data_list[train_len + val_len:]),
                   self.processed_paths[self.file_idx['test']])


class SocialGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, g2gc: Optional[Graph2GraphPairCompress]=None):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {'train': SocialGraphDataset(dataset_name=self.cfg.dataset.name,
                                                split='train', root=root_path,
                                                num_graphs=self.cfg.general.num_graphs,
                                                min_nodes=self.cfg.general.min_nodes,
                                                max_nodes=self.cfg.general.max_nodes,
                                                g2gc=g2gc),
                    'val': SocialGraphDataset(dataset_name=self.cfg.dataset.name,
                                              split='val', root=root_path,
                                              num_graphs=self.cfg.general.num_graphs,
                                              min_nodes=self.cfg.general.min_nodes,
                                              max_nodes=self.cfg.general.max_nodes,
                                              g2gc=g2gc),
                    'test': SocialGraphDataset(dataset_name=self.cfg.dataset.name,
                                               split='test', root=root_path,
                                               num_graphs=self.cfg.general.num_graphs,
                                               min_nodes=self.cfg.general.min_nodes,
                                               max_nodes=self.cfg.general.max_nodes,
                                               g2gc=g2gc)}

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

        if g2gc:
            self.train_dataset_orig = SocialGraphDataset(dataset_name=self.cfg.dataset.name,
                                                         split='train', root=root_path,
                                                         num_graphs=self.cfg.general.num_graphs,
                                                         min_nodes=self.cfg.general.min_nodes,
                                                         max_nodes=self.cfg.general.max_nodes,
                                                         g2gc=None)
            self.val_dataset_orig = SocialGraphDataset(dataset_name=self.cfg.dataset.name,
                                                       split='val', root=root_path,
                                                       num_graphs=self.cfg.general.num_graphs,
                                                       min_nodes=self.cfg.general.min_nodes,
                                                       max_nodes=self.cfg.general.max_nodes,
                                                       g2gc=None)
            self.test_dataset_orig = SocialGraphDataset(dataset_name=self.cfg.dataset.name,
                                                        split='test', root=root_path,
                                                        num_graphs=self.cfg.general.num_graphs,
                                                        min_nodes=self.cfg.general.min_nodes,
                                                        max_nodes=self.cfg.general.max_nodes,
                                                        g2gc=None)

    def __getitem__(self, item):
        return self.inner[item]


class SocialDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
