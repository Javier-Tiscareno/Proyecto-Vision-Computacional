"""
train_dcgan_maceta.py
---------------------------------------------------------
Autor: Javier Alvarado
Descripción:
    Entrena una red DCGAN para generar imágenes sintéticas de la
    clase "maceta". Utiliza parches de 64x64 preprocesados y sigue
    la arquitectura estándar propuesta en el paper original de DCGAN.

    Aunque en el proyecto final no se usaron GANs para augmentación
    debido a problemas de estabilidad y resultados poco convincentes,
    este script documenta el proceso experimental desarrollado.

Flujo del script:
    1. Cargar dataset de parches de macetas.
    2. Definir Generator y Discriminator (arquitecturas DCGAN).
    3. Inicializar parámetros y optimizadores.
    4. Entrenar por NUM_EPOCHS con la rutina típica GAN.
    5. Guardar imágenes generadas y pesos del modelo periódicamente.

Dependencias:
    - PyTorch
    - Torchvision
    - PIL
    - glob, os
---------------------------------------------------------
"""

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ============================================================
# CONFIGURACIÓN DEL ENTRENAMIENTO
# ============================================================

DATA_DIR = "/home/javier/Javier_Alvarado_Proyecto/gan_patches/maceta"
OUT_MODELS = "/home/javier/Javier_Alvarado_Proyecto/gan_models"
OUT_SAMPLES = "/home/javier/Javier_Alvarado_Proyecto/gan_samples/maceta"

IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 64
NUM_EPOCHS = 40         # Maceta tiene suficientes muestras; menos épocas son suficientes
LR = 2e-4
BETA1 = 0.5

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_SAMPLES, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# ============================================================
# DATASET: LECTOR DE PARCHES PARA GAN
# ============================================================

class PatchDataset(Dataset):
    """
    Lee parches de imágenes desde una carpeta y aplica transformaciones:
        - Resize al tamaño esperado por DCGAN
        - Conversión a tensor
        - Normalización a rango [-1,1]
    """

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
print(f"Imágenes para GAN (maceta): {len(dataset)}")


# ============================================================
# MODELOS: GENERADOR Y DISCRIMINADOR (DCGAN)
# ============================================================

class Generator(nn.Module):
    """
    Generador DCGAN:
    ConvTranspose2d → BatchNorm → ReLU, expandiendo hasta 64x64.
    """

    def __init__(self, latent_dim=100, ngf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Normalización a [-1,1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    Discriminador DCGAN:
    Conv2d → LeakyReLU → BatchNorm, terminando en probabilidad [0,1].
    """

    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


# Inicialización recomendada para GANs
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Instanciar modelos
G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)

G.apply(weights_init)
D.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))

# Ruido fijo para monitorear progreso
fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)


# ============================================================
# LOOP PRINCIPAL DE ENTRENAMIENTO
# ============================================================

for epoch in range(1, NUM_EPOCHS + 1):

    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)

        # Entrenamiento del discriminador ---------------------
        optimizerD.zero_grad()

        b_size = real_imgs.size(0)
        labels_real = torch.full((b_size,), 0.9, device=device)  # Label smoothing
        labels_fake = torch.zeros(b_size, device=device)

        output_real = D(real_imgs)
        lossD_real = criterion(output_real, labels_real)

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
        fake_imgs = G(noise)
        output_fake = D(fake_imgs.detach())
        lossD_fake = criterion(output_fake, labels_fake)

        lossD = lossD_real + lossD_fake
        optimizerD.step()

        # Entrenamiento del generador -------------------------
        optimizerG.zero_grad()
        output_fake_for_G = D(fake_imgs)
        lossG = criterion(output_fake_for_G, labels_real)  # Queremos que engañe al D
        lossG.backward()
        optimizerG.step()

    print(f"[Epoch {epoch}/{NUM_EPOCHS}] LossD: {lossD.item():.4f}  LossG: {lossG.item():.4f}")

    # Guardar cada 10 épocas
    if epoch % 10 == 0 or epoch == NUM_EPOCHS:
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()

        fake = (fake + 1) / 2  # Escalar a [0,1]

        import torchvision.utils as vutils
        out_path = os.path.join(OUT_SAMPLES, f"maceta_epoch_{epoch:03d}.png")
        vutils.save_image(fake, out_path, nrow=4)
        print(f"Guardado de muestras: {out_path}")

        model_path = os.path.join(OUT_MODELS, f"G_maceta_epoch_{epoch:03d}.pt")
        torch.save(G.state_dict(), model_path)
        print(f"Guardado del modelo: {model_path}")
