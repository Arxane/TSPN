import torch
import torch.nn.functional as F


def chamfer_distance_smoothed(point_set_a, point_set_b, sizes):
    point_set_a = point_set_a.float()
    point_set_b = point_set_b.float()

    B, N, D = point_set_a.shape
    _, M, _ = point_set_b.shape

    a = point_set_a.unsqueeze(2).expand(B, N, M, D)  # [B, N, M, D]
    b = point_set_b.unsqueeze(1).expand(B, N, M, D)  # [B, N, M, D]

    huber_dist = F.smooth_l1_loss(a, b, reduction='none').sum(-1)  # [B, N, M]

    setwise_distance = []
    for i in range(B):
        valid_n = sizes[i].item()
        valid_m = sizes[i].item()

        dist = huber_dist[i, :valid_n, :valid_m]
        min_a_to_b, _ = torch.min(dist, dim=-1)
        min_b_to_a, _ = torch.min(dist, dim=-2)

        mean_dist = (min_a_to_b.mean() + min_b_to_a.mean())
        setwise_distance.append(mean_dist)

    return torch.stack(setwise_distance)


if __name__ == '__main__':
    set_a = torch.tensor([
        [[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.]],
        [[2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]]
    ])

    set_b = torch.tensor([
        [[8.], [7.], [6.], [5.], [4.], [3.], [2.], [1.]],
        [[8.], [7.], [6.], [5.], [4.], [3.], [2.], [1.]]
    ])

    sizes = torch.tensor([5, 6])
    dist = chamfer_distance_smoothed(set_a, set_b, sizes)
    print("Chamfer Distances:", dist)
