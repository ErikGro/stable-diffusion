import torch
from torchvision.transforms import v2

from ldm.models.autoencoder import AutoencoderKL, VQModel
from PIL import Image
import yaml
from torchmetrics.image import StructuralSimilarityIndexMeasure

toImage = v2.Compose([
    v2.ToDtype(torch.uint8, scale=True),
    v2.ToPILImage()
])

transforms = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop((256, 256), scale=(1/4,1/4), ratio=(1,1)),
    v2.ToDtype(torch.float32, scale=True)
])

image = Image.open("he.jpg")
transformedImage = transforms(image)
toImage(transformedImage).save("cropped_input.png","PNG")

normalizedImage = v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(transformedImage)
batch = torch.unsqueeze(normalizedImage, 0)

ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

def encodeDecodeWithModel(name):
    modelPath = f"models/first_stage_models/{name}/model.ckpt"
    configPath = f"models/first_stage_models/{name}/config.yaml"
    config = {}
    with open(configPath) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ddconfig = config["model"]["params"]["ddconfig"]
    lossconfig = config["model"]["params"]["lossconfig"]
    embed_dim = config["model"]["params"]["embed_dim"]

    if name.startswith("vq"):
        autoencoder = VQModel(ddconfig, lossconfig, embed_dim=embed_dim, n_embed=config["model"]["params"]["n_embed"], ckpt_path=modelPath)
    else:
        autoencoder = AutoencoderKL(ddconfig, lossconfig, embed_dim, ckpt_path=modelPath)

    if name.startswith("vq"):
        latents = autoencoder.encode(batch)[0]
    else:
        latents = autoencoder.encode(batch).sample()

    encodedDecoded = autoencoder.decode(latents).squeeze()
    encodedDecoded = (encodedDecoded / 2 + 0.5).clamp(0, 1)

    encodedDecodedImage = toImage(encodedDecoded)

    ssim_val = ssim(transformedImage.unsqueeze(0), encodedDecoded.unsqueeze(0)).detach().item()

    encodedDecodedImage.save(f"{name}_ssim_{str(round(ssim_val, 2))}_dim_{latents.nelement()}.png", "PNG")


modelNames = [
    "kl-f4",
    "kl-f8",
    "vq-f4",
    "vq-f8"
]


for name in modelNames:
    encodeDecodeWithModel(name)
