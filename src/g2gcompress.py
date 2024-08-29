from typing import Optional, Union, cast

import torch
import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data


class CompressedData(Data):
    pass


class Graph2GraphPairCompress:
    def encode(self, data: Data) -> CompressedData:
        x = cast(Tensor, data.x)
        edge_index = cast(Tensor, data.edge_index)
        num_nodes = cast(int, data.num_nodes)

        edge_node_dists = torch.abs(x[edge_index[0]] - x[edge_index[1]]).sum(-1)
        edge_node_dists_argsorted = torch.argsort(edge_node_dists)
        node_to_pair = torch.full(
            (num_nodes,), -1, dtype=torch.int, device=x.device, requires_grad=False)
        node_is_higher = node_to_pair == 0
        num_pairs = 0
        working_edges = edge_index[:, edge_node_dists_argsorted]
        selected_edges = working_edges.new_zeros((2, 0), requires_grad=False)

        # Find the edges around which to create pairs
        while working_edges.numel():
            edge = working_edges[:, 0]
            selected_edges = torch.cat((selected_edges, edge.unsqueeze(dim=1)), dim=1)
            working_edges = working_edges[:, torch.isin(working_edges, edge, invert=True).all(dim=0)]

        num_pairs = selected_edges.size(dim=1)
        pair_numbers = torch.arange(
            0, num_pairs, dtype=torch.int, device=x.device, requires_grad=False)
        # Fill in bookkeeping information based on selected pair edges
        node_to_pair[selected_edges[0]] = node_to_pair[selected_edges[1]] = pair_numbers
        selected_edges_max, _selected_edges_max_indices = selected_edges.max(dim=0)
        node_is_higher[selected_edges_max] = True
        # Gather the single nodes
        single_node_mask = node_to_pair == -1
        num_single_nodes = cast(int, single_node_mask.sum().item())
        num_nodes_enc = num_pairs + num_single_nodes
        # Add one-hot encoding for pair/single nodes
        x_enc = x.new_zeros((num_nodes_enc, 2))
        x_enc[:num_pairs, 0] = 1.
        x_enc[-num_single_nodes:, 1] = 1.
        # Assign numbering to single nodes
        single_node_offsets = torch.arange(
            num_single_nodes, dtype=torch.int, device=x.device, requires_grad=False)
        node_to_pair[single_node_mask] = num_pairs + single_node_offsets
        # Map edges endpoints to their pair number
        edges_as_pairs = node_to_pair[edge_index]
        # Get mask for the edges between different pairs
        intra_pair_edges_mask = edges_as_pairs[0] != edges_as_pairs[1]
        edge_index = edge_index[:, intra_pair_edges_mask]
        edges_as_pairs = edges_as_pairs[:, intra_pair_edges_mask]
        # Map adjacency matrix indices to flat counterpart
        adj_mat_enc_flat_indices = edges_as_pairs[0] * num_nodes_enc + edges_as_pairs[1]
        # Fill edge value into adjacency matrix
        values_to_add = (
            ~(node_is_higher[edge_index[0]] | node_is_higher[edge_index[1]]) +
            (~node_is_higher[edge_index[0]] & node_is_higher[edge_index[1]]) * 2 +
            (node_is_higher[edge_index[0]] & ~node_is_higher[edge_index[1]]) * 4 +
            (node_is_higher[edge_index[0]] & node_is_higher[edge_index[1]]) * 8
        ).char()
        # Retrieve adjacency matrix from flattened version
        adj_mat_enc_flat = torch.zeros(
            num_nodes_enc * num_nodes_enc,
            dtype=torch.int8, device=x.device, requires_grad=False)
        adj_mat_enc_flat = adj_mat_enc_flat.index_add(0, adj_mat_enc_flat_indices, values_to_add)
        adj_mat_enc = adj_mat_enc_flat.view(num_nodes_enc, num_nodes_enc)
        # Fill compressed edges and metadata
        edge_index_enc = adj_mat_enc.nonzero().T
        edge_attr = adj_mat_enc[edge_index_enc[0], edge_index_enc[1]]

        # Only keep asc edges and mirror results for desc ones
        edge_mask_asc = edge_index_enc[0] < edge_index_enc[1]
        edge_index_enc = edge_index_enc[:, edge_mask_asc]
        edge_index_enc = torch.cat((edge_index_enc, edge_index_enc.flip(0)), dim=1)
        edge_attr = edge_attr[edge_mask_asc]
        edge_attr = edge_attr.repeat(2)

        cd_kwargs = data.to_dict()
        cd_kwargs.update(
            x=x_enc, edge_index=edge_index_enc, edge_attr=F.one_hot(edge_attr.long(), 16).float()
        )

        return CompressedData(**cd_kwargs)

    def decode(self, data: CompressedData) -> Data:
        x_enc: Tensor = data.x
        edge_index_enc: Tensor = data.edge_index
        edge_attr = data.edge_attr
        num_nodes_enc = cast(int, data.num_nodes)
        num_edges_enc = cast(int, data.num_edges)
        num_single_nodes = int(x_enc.sum())

        # Fill node features with constant information
        num_nodes_dec = 2 * x_enc.size(0) - num_single_nodes
        x_dec = x_enc.new_zeros((num_nodes_dec, 1))
        # Map from pair to first node in pair and update single node mapping
        first_single_node_enc = num_nodes_enc - num_single_nodes
        # Only work with asc edges and mirror results for desc ones later
        mask = edge_index_enc[0] < edge_index_enc[1]
        edge_index_enc = edge_index_enc[:, mask]
        num_edges_enc //= 2
        edge_attr = edge_attr[mask]
        edge_index_pair_head = edge_index_enc * 2 - \
            (edge_index_enc - first_single_node_enc).clamp(min=0)
        # Expand edges to all possible options
        edge_attr_offset_possible = torch.tensor(
            [[0, 0, 1, 1], [0, 1, 0, 1]],
            dtype=torch.int8, device=x_enc.device, requires_grad=False
        ).repeat(1, num_edges_enc)
        edge_index_possible = edge_index_pair_head.repeat_interleave(4, dim=1) + \
            edge_attr_offset_possible
        # Find mask of valid edges from all possible ones
        edge_attr_possible = edge_attr.repeat_interleave(4, dim=0)
        bit_position_possible = torch.tensor(
            [1, 2, 4, 8],
            dtype=torch.int8, device=x_enc.device, requires_grad=False
        ).repeat(num_edges_enc)
        edge_mask_possible = edge_attr_possible.bitwise_and(bit_position_possible).bool()
        # Get intra-pair nodes and put them together with inter-pair ones
        edge_index_inter = edge_index_possible[:, edge_mask_possible]
        edge_index_intra_asc = torch.arange(
            0, 2 * first_single_node_enc,
            dtype=torch.int, device=x_enc.device, requires_grad=False).view(-1, 2).T
        edge_index_dec = torch.cat(
            [
                edge_index_inter, edge_index_intra_asc,
                edge_index_inter.flip(0), edge_index_intra_asc.flip(0)
            ],dim=1)

        d_kwargs = data.to_dict()
        d_kwargs.update(
            x=x_dec, edge_index=edge_index_dec,
            edge_attr=F.one_hot(edge_index_dec.new_ones(edge_index_dec.size(dim=1))).float())

        return Data(**d_kwargs)
