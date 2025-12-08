"""
train_dcgan_cubeta.py
---------------------------------------------------------
Autor: Javier Alvarado
Descripción:
    Entrena una red GAN del tipo DCGAN para generar imágenes
    sintéticas de la clase "cubeta". El entrenamiento utiliza
    parches de 128x128 previamente extraídos del dataset.

    Aunque en el proyecto final se descartó el uso de GAN por
    problemas de estabilidad y calidad, este script documenta
    completamente el proceso experimental original.

Flujo del script:
    1. Cargar el dataset de parches de cubetas.
    2. Definir arquitecturas Generator y Discriminator (DCGAN).
    3. Inicializar pesos y optimizadores (Adam).
    4. Entrenar la GAN mediante alternancia:
        - Entrenamiento del discriminador
        - Entrenamiento del generador
    5. Guardar muestras generadas y pesos del modelo.

Dependencias:
    - PyTorch
    - Torchvision
    - PIL
    - OpenCV (opcional)
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

DATA_DIR = "/home/javier/Javier_Alvarado_Proyecto/gan_patches/cubeta"
OUT_MODELS = "/home/javier/Javier_Alvarado_Proyecto/gan_models"
OUT_SAMPLES = "/home/javier/Javier_Alvarado_Proyecto/gan_samples/cubeta"

IMG_SIZE = 128
LATENT_DIM = 100
BATCH_SIZE = 64
NUM_EPOCHS = 150     # Ajustable según tiempo disponible
LR = 2e-4            # Learning rate recomendado para DCGAN
BETA1 = 0.5          # Parámetro típico de Adam para GANs

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_SAMPLES, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# ============================================================
# DATASET PERSONALIZADO
# ============================================================

class PatchDataset(Dataset):
    """
    Dataset que carga parches de imágenes desde una carpeta.
    Las imágenes se redimensionan y normalizan según DCGAN.
    """

    def __init__(self, folder, img_size=128):
        exts = (".jpg", ".jpeg", ".png")
        self.files = []

        # Buscar imágenes con extensiones soportadas
        for e in exts:
            self.files.extend(glob.glob(os.path.join(folder, f"*{e}")))
        if len(self.files) == 0:
            raise ValueError(f"No se encontraron imágenes en {folder}")

        # Transformaciones estándar
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]  # Normalización a rango [-1,1]
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
print(f"Imágenes disponibles para GAN (cubeta): {len(dataset)}")


# ============================================================
# MODELOS DCGAN: GENERADOR Y DISCRIMINADOR
# ============================================================

class Generator(nn.Module):
    """
    Generador DCGAN:
    ConvTranspose → BatchNorm → ReLU hasta formar imagen 64x64.
    """

    def __init__(self, latent_dim=100, ngf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            # Z (latent_dim) → (ngf*8) x 4 x 4
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 4x4 → 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 8x8 → 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 16x16 → 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 32x32 → 64x64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Salida entre [-1,1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    Discriminador DCGAN:
    Conv → LeakyReLU → BatchNorm → salida sigmoide (real/falso).
    """

    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # Entrada 64x64
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


# Inicializar pesos como recomienda el paper DCGAN
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)

G.apply(weights_init)
D.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))

# Rui
