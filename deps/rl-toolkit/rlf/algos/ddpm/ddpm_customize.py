# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# matplotlib inline
import torch
import argparse
import numpy as np
import torch
import torch.nn as nn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


########### hyper parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_steps = 1000
batch_size = 128  # 128
num_epoch = 10000

# decide beta
betas = torch.linspace(-6, 6, num_steps).to(device)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

# calculate alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
       alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
       == one_minus_alphas_bar_sqrt.shape
# print("all the same shape", betas.shape)


########### decide the sample during definite diffusion process
# calculate x on given time based on x_0 and re-parameterization
def q_x(x_0, t):
    """based on x[0], get x[t] on any given time t"""
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)  # adding noise based on x[0]在x[0]


########### gaussian distribution in reverse diffusion process


class MLPDiffusionCustomize(nn.Module):
    def __init__(self, n_steps, input_dim=6, num_units=128, depth=4, device='cuda'):
        super(MLPDiffusionCustomize, self).__init__()
        linears_list = []
        linears_list.append(nn.Linear(input_dim, num_units))
        linears_list.append(nn.ReLU())
        if depth > 1:
            for i in range(depth - 1):
                linears_list.append(nn.Linear(num_units, num_units))
                linears_list.append(nn.ReLU())
        linears_list.append(nn.Linear(num_units, input_dim))
        self.linears = nn.ModuleList(linears_list).to(device)

        embed_list = []
        for i in range(depth - 1):
            embed_list.append(nn.Embedding(n_steps, num_units))
        if depth == 1:
            embed_list.append(nn.Embedding(n_steps, num_units))
        self.step_embeddings = nn.ModuleList(embed_list).to(device)

    def forward(self, x, t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x


def norm_vec(x, mean, std):
    obs_x = torch.clamp((x - mean)
                        / (std + 1e-8),
                        -10.0,
                        10.0,
                        )
    return obs_x


########### training loss funciton
# sample at any given time t, and calculate sampling loss
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]

    # generate eandom t for a batch data
    t = torch.randint(0, n_steps, size=(batch_size // 2,)).to(device)
    t = torch.cat([t, n_steps - 1 - t], dim=0)  # [batch_size, 1]
    t = t.unsqueeze(-1)

    # coefficient of x0
    a = alphas_bar_sqrt[t]

    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]

    # generate random noise eps
    e = torch.randn_like(x_0)

    # model input
    x = x_0 * a + e * aml

    # get predicted randome noise at time t
    output = model(x, t.squeeze(-1))

    # calculate the loss between actual noise and predicted noise
    return (e - output).square().mean()


########### reverse diffusion sample function（inference）
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    # generate[T-1]、x[T-2]|...x[0] from x[T]
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    # sample reconstruction data at time t drom x[T]
    t = torch.tensor([t]).to(device)

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x, t)

    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z

    return (sample)


def reconstruct(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    # generate random t for a batch data
    t = torch.ones_like(x_0, dtype=torch.long).to(device) * n_steps

    # coefficient of x0
    a = alphas_bar_sqrt[t]

    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]

    # generate random noise eps
    e = torch.randn_like(x_0).to(device)

    # model input
    x_T = x_0 * a + e * aml

    # generate[T-1]、x[T-2]|...x[0] from x[T]
    for i in reversed(range(n_steps)):
        x_T = p_sample(model, x_T, i, betas, one_minus_alphas_bar_sqrt)
    x_construct = x_T

    return x_construct


########### start training, print loss and print the medium reconstrction result
seed = 1234


class EMA():  # Exponential Moving Average
    # EMA
    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average