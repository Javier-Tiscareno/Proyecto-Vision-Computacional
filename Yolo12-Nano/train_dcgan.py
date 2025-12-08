import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils


# ---------- MODELO DCGAN ----------

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # input Z: (nz) -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf) x 32 x 32 -> (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # (nc) x 64 x 64 -> (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 32 x 32 -> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8) x 4 x 4 -> 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------- ENTRENAMIENTO ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Carpeta con los parches de una clase (ej. gan_patches/cubeta)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Carpeta donde guardar checkpoints y muestras")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nz", type=int, default=100, help="Dimensión del ruido")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transformaciones: escalado [0,1] -> normalización [-1,1]
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(
        root=str(data_dir.parent),  # el padre debe contener subcarpeta con el nombre de clase
        transform=transform,
    )

    # Filtrar solo la clase correspondiente
    # (ImageFolder asigna índices de clase por orden alfabético)
    class_idx = sorted(os.listdir(data_dir.parent)).index(data_dir.name)
    indices = [i for i, (_, y) in enumerate(dataset.samples) if y == class_idx]
    print(f"Parches usados para {data_dir.name}: {len(indices)}")

    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    nz = args.nz
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0

    step = 0
    for epoch in range(args.epochs):
        for i, (data, _) in enumerate(dataloader):
            # --- Actualizar D ---
            netD.zero_grad()
            real = data.to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # --- Actualizar G ---
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch+1}/{args.epochs}][{i}/{len(dataloader)}] "
                      f"Loss_D: {(errD_real+errD_fake).item():.4f} Loss_G: {errG.item():.4f}")
            step += 1

        # Guardar muestras de la época
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        utils.save_image(
            fake,
            out_dir / f"fake_samples_epoch_{epoch+1:03d}.png",
            normalize=True,
            nrow=8,
        )

        # Guardar modelos
        torch.save(netG.state_dict(), out_dir / f"netG_epoch_{epoch+1:03d}.pth")
        torch.save(netD.state_dict(), out_dir / f"netD_epoch_{epoch+1:03d}.pth")

    print("Entrenamiento DCGAN terminado.")


if __name__ == "__main__":
    main()
