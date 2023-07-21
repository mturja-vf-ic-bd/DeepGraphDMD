import torch
import torch.nn as nn

import unittest


def get_neighbor_edge_indices(n, edge):
    src, dst = edge
    neigh_edges = set()
    for i in range(0, n):
        if i == src:
            continue
        if src < i:
            neigh_edges.add((src, i))
        else:
            neigh_edges.add((i, src))

    for i in range(0, n):
        if i == dst:
            continue
        if dst < i:
            neigh_edges.add((dst, i))
        else:
            neigh_edges.add((i, dst))
    neigh_edges = list(neigh_edges)
    index = neigh_edges.index((src, dst))
    return neigh_edges, index


def get_upper_triangular_indices(n):
    dummy = torch.ones((n, n))
    upr_ind = torch.where(torch.triu(dummy, diagonal=1) == 1)
    return upr_ind[0], upr_ind[1]


def get_neighbor_edge_matrix(n):
    src, dst = get_upper_triangular_indices(n)
    edge_list = []
    index_list = []
    for n1, n2 in zip(src, dst):
        edge_n, index = get_neighbor_edge_indices(n, (n1.item(), n2.item()))
        edge_list.append(edge_n)
        index_list.append(index)
    return torch.LongTensor(edge_list), torch.LongTensor(index_list)


def standardize_g(g):
    g = (g - g.mean(dim=1, keepdim=True)) / g.std(dim=1, keepdim=True)
    return g


def compute_window_lkis_loss(Z, w, N, stride):
    loss_rss = 0
    count = 0
    _, self_indices = get_neighbor_edge_matrix(N)
    self_indices = self_indices.to(Z.device)
    self_indices = self_indices.expand(Z.shape[0], w - 1, Z.shape[2]).permute(0, 2, 1).unsqueeze(-1)
    for i in range(0, Z.shape[1] - w, stride):
        start = i
        end = i + w
        g0 = Z[:, start:end - 1]   # (b, w-1, n(n-1)/2, m)
        g1 = Z[:, start + 1:end]
        g0 = g0.permute(0, 2, 1, 3)   # (b, n(n-1)/2, w-1, m)
        g1 = g1.permute(0, 2, 1, 3)
        cov = g0.transpose(2, 3) @ g0  # (b, n(n-1)/2, m, m)
        g0inv = torch.linalg.inv(cov) @ g0.transpose(2, 3)
        k = g0inv @ g1
        g1_pred = g0 @ k
        g1 = g1.gather(3, self_indices).squeeze(-1)
        g1_pred = g1_pred.gather(3, self_indices).squeeze(-1)
        g1 = standardize_g(g1)
        g1_pred = standardize_g(g1_pred)
        loss_rss += nn.MSELoss()(g1_pred, g1)
        count += 1
    return loss_rss / count


def compute_adjacency_pred_loss(G, A, k):
    top_k_ind = torch.topk(A.flatten(start_dim=2), k=k, dim=-1)[1]
    bottom_k_ind = torch.topk(-A.flatten(start_dim=2), k=k, dim=-1)[1]
    pos_pred = torch.gather(G.flatten(start_dim=2), dim=-1, index=top_k_ind)
    neg_pred = torch.gather(G.flatten(start_dim=2), dim=-1, index=bottom_k_ind)
    pred = torch.cat([pos_pred, neg_pred], dim=-1)
    pos_true = torch.gather(A.flatten(start_dim=2), dim=-1, index=top_k_ind)
    neg_true = torch.gather(A.flatten(start_dim=2), dim=-1, index=bottom_k_ind)
    true = torch.cat([pos_true, neg_true], dim=-1)
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred, true)
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss = loss_fn(torch.cat([pos_pred, neg_pred], dim=-1),
    #                torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=-1))
    return loss


def corr_coeff(A):
    eps = 1e-5
    m = A.mean(dim=-1, keepdim=True)
    s = A.std(dim=-1, keepdim=True)
    N = A.size(-1)
    A = A - m
    cov = (A @ A.transpose(-1, -2)) / (N - 1)
    corr = cov / (s @ s.transpose(-1, -2) + eps)
    return corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))


class testEdgeNeighborFunctions(unittest.TestCase):
    def testGetNeighborEdgeIndices(self):
        n = 5
        # Test case for edge (1, 3)
        computed_edges, _ = get_neighbor_edge_indices(n, (1, 3))
        actual_edges = {(0, 3), (1, 2), (2, 3), (0, 1), (1, 3), (3, 4), (1, 4)}
        self.assertEqual(len(computed_edges), 2 * n - 3)
        self.assertEqual(len(set(computed_edges) - actual_edges), 0)

        # Test case for edge(0, 4)
        computed_edges, _ = get_neighbor_edge_indices(n, (0, 4))
        actual_edges = {(0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4)}
        self.assertEqual(len(computed_edges), 2 * n - 3)
        self.assertEqual(len(set(computed_edges) - actual_edges), 0)

    def testGetNeighborEdgeMatrix(self):
        n = 5
        indices, self_indices = get_neighbor_edge_matrix(n)
        self.assertEqual(indices.size(), (n * (n - 1) // 2, 2 * n - 3, 2))

    def testSelfIndices(self):
        n = 5
        indices, self_indices = get_neighbor_edge_matrix(n)
        indices = indices[:, :, 0] * n + indices[:, :, 1]
        edge_ids = indices.gather(1, self_indices.unsqueeze(1)).squeeze(1)
        # compute original edge_ids from upper triangle
        upr_mat = []
        for i in range(n):
            for j in range(i + 1, n):
                upr_mat.append(i * n + j)
        upr_mat = torch.Tensor(upr_mat)
        self.assertTrue((upr_mat == edge_ids).all())

    def testGetNeighborEdgeMatrix2(self):
        n = 5
        indices, self_indices = get_neighbor_edge_matrix(n)
        flat_indices = indices[:, :, 0] * n + indices[:, :, 1]
        dummy = torch.randn((1, n * n)).repeat(n * (n - 1) // 2, 1)
        reduced_dummy = torch.gather(dummy, dim=1, index=flat_indices)
        self.assertEqual(reduced_dummy.size(), (n * (n - 1) // 2, 2 * n - 3))

        dummy2 = dummy[0]
        reduced_dummy2 = dummy2[flat_indices]
        self.assertTrue((reduced_dummy2 == reduced_dummy).all())

    def testCacheNeighborEdgeMatrix(self):
        n = 50
        indices, self_indices = get_neighbor_edge_matrix(n)
        with open(f"nem_{n}.pt", "wb") as f:
            torch.save(indices, f)
        with open(f"nem_{n}.pt", "rb") as f:
            loaded_indices = torch.load(f)
        self.assertTrue((indices == loaded_indices).all())
