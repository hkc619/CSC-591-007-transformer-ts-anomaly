"""保留官方 solver 的 KL、minimax objective 與 temperature=50 native energy。"""
import torch
import torch.nn.functional as F


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def normalized_prior(prior):
    return prior / prior.sum(dim=-1, keepdim=True)


def training_objectives(inputs, outputs, series, priors, k=3):
    series_loss, prior_loss = 0.0, 0.0
    for s, prior in zip(series, priors):
        p = normalized_prior(prior)
        series_loss = series_loss + (my_kl_loss(s, p.detach()).mean() + my_kl_loss(p.detach(), s).mean())
        prior_loss = prior_loss + (my_kl_loss(p, s.detach()).mean() + my_kl_loss(s.detach(), p).mean())
    series_loss, prior_loss = series_loss / len(priors), prior_loss / len(priors)
    reconstruction = F.mse_loss(outputs, inputs)
    return reconstruction - k * series_loss, reconstruction + k * prior_loss, reconstruction, series_loss, prior_loss


def native_energy(inputs, outputs, series, priors, temperature=50):
    reconstruction = (inputs - outputs).square().mean(dim=-1)
    series_loss, prior_loss = 0.0, 0.0
    for s, prior in zip(series, priors):
        p = normalized_prior(prior)
        series_loss = series_loss + my_kl_loss(s, p.detach()) * temperature
        prior_loss = prior_loss + my_kl_loss(p, s.detach()) * temperature
    # Official test-time formula sums layers (does not divide by layer count).
    return torch.softmax(-series_loss - prior_loss, dim=-1) * reconstruction
