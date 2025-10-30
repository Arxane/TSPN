import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import matplotlib.pyplot as plt

class SetPrior(nn.Module):
    def __init__(self, event_size):
        super(SetPrior, self).__init__()
        self.event_size = event_size
        self.loc = nn.Parameter(torch.zeros(event_size))
        self.log_scale = nn.Parameter(torch.zeros(event_size))

    def forward(self, batch_size):
        if isinstance(batch_size, torch.Tensor):
            batch_size = int(batch_size.item())
        else:
            batch_size = int(batch_size)

        loc = self.loc.unsqueeze(0).expand(batch_size, -1)
        scale = self.log_scale.exp().unsqueeze(0).expand(batch_size, -1)
        return torch.distributions.Normal(loc, scale)


    
if __name__ == "__main__":
    batch_size = 1000
    event_size = 2

    prior = SetPrior(event_size)
    mvn = prior(batch_size)
    samples = mvn.sample()

    plt.scatter(samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy(), alpha=0.6)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()