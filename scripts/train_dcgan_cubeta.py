import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATA_DIR = "/home/javier/Javier_Alvarado_Proyecto/gan_patches/cubeta"
OUT_MODELS = "/home/javier/Javier_Alvarado_Proyecto/gan_models"
OUT_SAMPLES = "/home/javier/Javier_Alvarado_Proyecto/gan_samples/cubeta"

IMG_SIZE = 128
LATENT_DIM = 100
BATCH_SIZE = 64
NUM_EPOCHS = 150       # puedes bajarlo a 80 si ves que se tarda mucho
LR = 2e-4
BETA1 = 0.5

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_SAMPLES, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando dispositivo: {device}")


# ============================================================
# DATASET
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, folder, img_size=128):
        exts = (".jpg", ".jpeg", ".png")
        self.files = []
        for e in exts:
            self.files.extend(glob.glob(os.path.join(folder, f"*{e}")))
        if len(self.files) == 0:
            raise ValueError(f"No se encontraron imágenes en {folder}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img


dataset = PatchDataset(DATA_DIR, IMG_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f"[INFO] Imágenes para GAN (cubeta): {len(dataset)}")


# ============================================================
# MODELOS DCGAN
# ============================================================

class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            # input Z: (latent_dim) → (ngf*8) x 4 x 4
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8) x 4 x 4 → (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4) x 8 x 8 → (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2) x 16 x 16 → (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf) x 32 x 32 → nc x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # nc x 64 x 64 → (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 32 x 32 → (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 16 x 16 → (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x 8 x 8 → (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8) x 4 x 4 → 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)

G.apply(weights_init)
D.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))

fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)


# ============================================================
# LOOP DE ENTRENAMIENTO
# ============================================================

for epoch in range(1, NUM_EPOCHS + 1):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)

        # --------------------------------
        #  Entrenar Discriminador
        # --------------------------------
        optimizerD.zero_grad()

        b_size = real_imgs.size(0)
        labels_real = torch.ones(b_size, device=device)
        labels_fake = torch.zeros(b_size, device=device)

        output_real = D(real_imgs)
        lossD_real = criterion(output_real, labels_real)

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
        fake_imgs = G(noise)
        output_fake = D(fake_imgs.detach())
        lossD_fake = criterion(output_fake, labels_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # --------------------------------
        #  Entrenar Generador
        # --------------------------------
        optimizerG.zero_grad()
        output_fake_for_G = D(fake_imgs)
        lossG = criterion(output_fake_for_G, labels_real)  # queremos que D se "engañe"
        lossG.backward()
        optimizerG.step()

    print(f"[Epoch {epoch}/{NUM_EPOCHS}] LossD: {lossD.item():.4f}  LossG: {lossG.item():.4f}")

    # Guardar muestras y modelo cada 10 epochs
    if epoch % 10 == 0 or epoch == NUM_EPOCHS:
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        # Desnormalizar de [-1,1] a [0,1]
        fake = (fake + 1) / 2
        import torchvision.utils as vutils
        out_path = os.path.join(OUT_SAMPLES, f"cubeta_epoch_{epoch:03d}.png")
        vutils.save_image(fake, out_path, nrow=4)
        print(f"    [Guardado] Muestras en {out_path}")

        model_path = os.path.join(OUT_MODELS, f"G_cubeta_epoch_{epoch:03d}.pt")
        torch.save(G.state_dict(), model_path)
        print(f"    [Guardado] Generador en {model_path}")
