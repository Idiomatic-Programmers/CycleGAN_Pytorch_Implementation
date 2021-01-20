import torch


def gp(discriminator, real, fake, device="cpu"):

    BATCH_SIZE, C, H, W = real.size()
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1).repeat(1, C, H, W).to(device)
    interpolated = real * epsilon + fake * (1 - epsilon)

    mixed_score = discriminator(interpolated)
    gradient = torch.autograd.grad(inputs=interpolated, outputs=mixed_score, grad_outputs=torch.ones_like(mixed_score),
                                   create_graph=True, retain_graph=True)[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penality = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penality
