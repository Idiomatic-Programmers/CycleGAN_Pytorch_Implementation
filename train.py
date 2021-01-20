from torchvision import transforms
from datasets.data import get_image_data_loader
import torch
import os

import torchvision

from models.modules import Generator, Discriminator
from models.utils import gp

from torch.utils.tensorboard import SummaryWriter

config = {
    'channels': 3,
    'epoch': 200,
    'generator_lr': 0.005,
    'discriminator_lr': 0.0002,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch': 1,
    'lambda': 10.0,
    'lambda_gp': 10.0,
    'identity_lambda': 50.0,
    'weights_clip': 0.01,
    'discriminator_iterations': 5,
    'load_state': 'latest',
    'checkpoint_path': 'checkpoints'
}

print("Training using", config['device'])

transformations = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train = get_image_data_loader(os.path.join("datasets", "img2ai", "trainA"),
                              os.path.join("datasets", "img2ai", "trainB"),
                              transformations, transformations, batch=config['batch'])

G = Generator(config['channels']).to(config['device'])
F = Generator(config['channels']).to(config['device'])
Dx = Discriminator(config['channels']).to(config['device'])
Dy = Discriminator(config['channels']).to(config['device'])

try:
    G.load_state_dict(torch.load(os.path.join(config['checkpoint_path'], f'G_{config["load_state"]}.pth')))
    F.load_state_dict(torch.load(os.path.join(config['checkpoint_path'], f'F_{config["load_state"]}.pth')))
    Dx.load_state_dict(torch.load(os.path.join(config['checkpoint_path'], f'Dx_{config["load_state"]}.pth')))
    Dy.load_state_dict(torch.load(os.path.join(config['checkpoint_path'], f'Dy_{config["load_state"]}.pth')))
    print("Loaded weights from checkpoints.")
except:
    print("No checkpoints found, initialised new weights.")

identity_loss = torch.nn.MSELoss()

step = 0

fake_writer = SummaryWriter('logs/fake', comment='Generated Illustration Image')
photo_writer = SummaryWriter('logs/real', comment='Image')

optimiser_g = torch.optim.Adam(params=G.parameters(), lr=config['generator_lr'], betas=(0.55, 0.9))
optimiser_f = torch.optim.Adam(params=F.parameters(), lr=config['generator_lr'], betas=(0.55, 0.9))
optimiser_dx = torch.optim.Adam(params=Dx.parameters(), lr=config['discriminator_lr'], betas=(0.55, 0.9))
optimiser_dy = torch.optim.Adam(params=Dy.parameters(), lr=config['discriminator_lr'], betas=(0.55, 0.9))

for epoch in range(config['epoch']):
    for batch, (X, Y) in enumerate(train):

        X = X.to(config['device'])
        Y = Y.to(config['device'])

        # TRAIN DISCRIMINATORS
        for _ in range(config['discriminator_iterations']):
            y_hat = G(X)

            score_y_hat = Dy(y_hat).reshape(-1)
            score_y = Dy(Y).reshape(-1)

            gradient_penality_Dy = gp(Dy, Y, y_hat, config['device'])
            loss_ad_forward = (- (torch.mean(score_y) - torch.mean(score_y_hat)) + config['lambda_gp'] * gradient_penality_Dy)

            optimiser_dy.zero_grad()
            loss_ad_forward.backward()
            optimiser_dy.step()

            x_hat = F(Y)

            score_x_hat = Dx(x_hat).reshape(-1)
            score_x = Dx(X).reshape(-1)

            gradient_penality_Dx = gp(Dx, X, x_hat, config['device'])
            loss_ad_backward = (- (torch.mean(score_x) - torch.mean(score_x_hat)) + config['lambda_gp'] * gradient_penality_Dx)

            optimiser_dx.zero_grad()
            loss_ad_backward.backward()
            optimiser_dx.step()

            # for p in Dx.parameters():
            #     p.data.clamp_(-config['weights_clip'], config['weights_clip'])
            #
            # for p in Dy.parameters():
            #     p.data.clamp_(-config['weights_clip'], config['weights_clip'])

        # TRAIN GENERATORS
        reconstructed_x = F(G(X))
        score_reconstructed_x = Dx(reconstructed_x).reshape(-1)
        identity_y = F(Y)

        cc_loss_forward = (config['lambda'] * (-torch.mean(score_reconstructed_x))
                           + config['identity_lambda'] * identity_loss(Y, identity_y))

        optimiser_f.zero_grad()
        cc_loss_forward.backward()
        optimiser_f.step()

        reconstructed_y = G(F(Y))
        score_reconstructed_y = Dy(reconstructed_y)
        identity_x = G(X)
        cc_loss_backward = (config['lambda'] * (-torch.mean(score_reconstructed_y))
                            + config['identity_lambda'] * identity_loss(X, identity_x))

        optimiser_g.zero_grad()
        cc_loss_backward.backward()
        optimiser_g.step()

        loss = loss_ad_forward + loss_ad_backward + cc_loss_forward + cc_loss_backward

        if step % 20 == 0:
            print(
                f"Epoch [{epoch}/{config['epoch']}] Batch {batch}/{len(train)} "
                f"Loss: {loss.item():.4f} Ad Loss (Forword): {loss_ad_forward.item():.4f} "
                f"Ad Loss (Backward): {loss_ad_backward.item():.4f} CC Loss (Forward): {cc_loss_forward.item():.4f} "
                f"CC Loss (Backward): {cc_loss_backward.item():.4f}"
            )

        if step % 100 == 0:
            with torch.no_grad():
                out = G(X)

                if not os.path.exists(config['checkpoint_path']):
                    os.mkdir(config['checkpoint_path'])

                torch.save(G.state_dict(), os.path.join(config['checkpoint_path'], f'G_{step}.pth'))
                torch.save(F.state_dict(), os.path.join(config['checkpoint_path'], f'F_{step}.pth'))
                torch.save(Dx.state_dict(), os.path.join(config['checkpoint_path'], f'Dx_{step}.pth'))
                torch.save(Dy.state_dict(), os.path.join(config['checkpoint_path'], f'Dy_{step}.pth'))

                torch.save(G.state_dict(), os.path.join(config['checkpoint_path'], f'G_latest.pth'))
                torch.save(F.state_dict(), os.path.join(config['checkpoint_path'], f'F_latest.pth'))
                torch.save(Dx.state_dict(), os.path.join(config['checkpoint_path'], f'Dx_latest.pth'))
                torch.save(Dy.state_dict(), os.path.join(config['checkpoint_path'], f'Dy_latest.pth'))

                img_grid_real = torchvision.utils.make_grid(
                    X[:config['batch']], normalize=True
                )

                img_grid_fake = torchvision.utils.make_grid(
                    out[:config['batch']], normalize=True
                )

                fake_writer.add_image("fake", img_grid_fake, global_step=step)
                photo_writer.add_image("real", img_grid_real, global_step=step)
        step += 1
    if epoch > config['epoch'] // 2:
        config['lr'] -= config['lr'] / (config['epoch'] - epoch)
        print(f"New LR : {config['lr']}")
