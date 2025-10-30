import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_gather(x, idx, dim):
    idx_expanded = idx.expand(*x.shape[:-1], idx.shape[-1])
    return torch.gather(x, dim, idx_expanded)


class FSEncoder(nn.Module):
    def __init__(self, in_channels, encoder_dim, encoder_output_channels, n_pieces):
        super(FSEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=encoder_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(encoder_dim, encoder_output_channels, kernel_size=1)

        self.pool = FSPool(encoder_output_channels, n_pieces)

        self.linear1 = nn.Linear(encoder_output_channels, encoder_output_channels)
        self.linear2 = nn.Linear(encoder_output_channels, encoder_output_channels)

    def forward(self, x, sizes):
        if x.shape[1] != self.conv1.in_channels:
            x = x.transpose(1, 2)  

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        x = x.transpose(1, 2)  
        x, perm = self.pool(x, sizes)

        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)

        return x


class FSPool(nn.Module):
    def __init__(self, in_channels, n_pieces, relaxed=False):
        super(FSPool, self).__init__()
        self.n_pieces = n_pieces
        self.relaxed = relaxed
        self.weight = nn.Parameter(torch.randn(in_channels, n_pieces + 1))

    def forward(self, x, n=None):
        assert x.shape[1] == self.weight.shape[0], "Incorrect number of input channels in weight"

        if n is None:
            n = torch.full((x.size(0),), x.size(2), dtype=torch.long, device=x.device)

        sizes, mask = fill_sizes(n, x)
        mask = mask.expand_as(x)

        weight = self.determine_weight(sizes)

        x = x + (1 - mask) * -99999
        if self.relaxed:
            sorted_x, perm = cont_sort(x, temp=self.relaxed)
        else:
            sorted_x, perm = torch.sort(x, dim=2, descending=True)

        x = torch.sum(sorted_x * weight * mask, dim=2)
        return x, perm

    def determine_weight(self, sizes):
        batch_size = sizes.size(0)
        in_channels, _ = self.weight.shape

        weight = self.weight.unsqueeze(0).expand(batch_size, in_channels, -1)

        index = self.n_pieces * sizes
        index = index.unsqueeze(1).expand(batch_size, in_channels, -1)

        idx = index.long()
        frac = index - idx.float()

        left = torch_gather(weight, idx, dim=2)
        right = torch_gather(
            weight,
            torch.clamp(idx + 1, min=0, max=self.n_pieces),
            dim=2
        )

        return (1 - frac) * left + frac * right


def fill_sizes(sizes, x=None):
    if x is not None:
        max_size = x.size(2)
    else:
        max_size = torch.max(sizes).item()

    size_tensor = torch.arange(max_size, dtype=torch.float32, device=sizes.device).unsqueeze(0)
    total_sizes = (sizes.float() - 1).clamp(min=1).unsqueeze(1)
    ratios = size_tensor / total_sizes

    mask = (ratios <= 1).float().unsqueeze(1)
    ratios = torch.clamp(ratios, min=0, max=1)

    return ratios, mask


def deterministic_sort(s, tau):
    n = s.size(1)
    one = torch.ones((n, 1), device=s.device)
    A_s = torch.abs(s - s.transpose(1, 2))
    B = A_s @ (one @ one.transpose(0, 1))
    scaling = torch.arange(n, device=s.device).float()
    scaling = (n + 1 - 2 * (scaling + 1)).view(1, 1, n)
    C = s @ scaling
    P_max = (C - B).transpose(1, 2)
    P_hat = P_max / tau
    return F.softmax(P_hat, dim=-1)


def cont_sort(x, perm=None, temp=1):
    original_size = x.size()
    x_flat = x.view(-1, x.size(2), 1)

    if perm is None:
        perm = deterministic_sort(x_flat, temp)
    else:
        perm = perm.transpose(1, 2)

    sorted_x = perm.transpose(1, 2) @ x_flat
    sorted_x = sorted_x.view(original_size)

    return sorted_x, perm
