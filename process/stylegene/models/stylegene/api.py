import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import os
from torchvision import transforms
from stylegene.models.stylegan2.model import Generator
from stylegene.models.encoders.psp_encoders import Encoder4Editing
from stylegene.models.stylegene.model import MappingSub2W, MappingW2Sub
from stylegene.models.stylegene.util import get_keys, requires_grad, load_img
from stylegene.models.stylegene.gene_pool import GenePoolFactory
from stylegene.models.stylegene.gene_crossover_mutation import fuse_latent
from stylegene.models.stylegene.fair_face_model import init_fair_model, predict_race
from stylegene.configs import path_ckpt_e4e, path_ckpt_stylegan2, path_ckpt_stylegene, path_ckpt_genepool, path_dataset_ffhq


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
USE_GENE_POOL = True if os.environ['USE_GENE_POOL'] == 'true' else False
MAX_SAMPLES = int(os.environ['MAX_SAMPLES'])


def init_model(image_size=1024, latent_dim=512):
    ckp = torch.load(path_ckpt_e4e, map_location='cpu')
    encoder = Encoder4Editing(50, 'ir_se', image_size).eval()
    encoder.load_state_dict(get_keys(ckp, 'encoder'), strict=True)
    mean_latent = ckp['latent_avg'].to('cpu')
    mean_latent.unsqueeze_(0)

    generator = Generator(image_size, latent_dim, 8)
    checkpoint = torch.load(path_ckpt_stylegan2, map_location='cpu')
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator.eval()
    sub2w = MappingSub2W(N=18).eval()
    w2sub34 = MappingW2Sub(N=18).eval()
    ckp = torch.load(path_ckpt_stylegene, map_location='cpu')
    w2sub34.load_state_dict(get_keys(ckp, 'w2sub34'))
    sub2w.load_state_dict(get_keys(ckp, 'sub2w'))

    requires_grad(sub2w, False)
    requires_grad(w2sub34, False)
    requires_grad(encoder, False)
    requires_grad(generator, False)
    return encoder, generator, sub2w, w2sub34, mean_latent


# init model
encoder, generator, sub2w, w2sub34, mean_latent = init_model()
encoder, generator, sub2w, w2sub34, mean_latent = encoder.to(device), generator.to(device), sub2w.to(
    device), w2sub34.to(device), mean_latent.to(device)
model_fair_7 = init_fair_model(device)  # init FairFace model

# load a GenePool
geneFactor = GenePoolFactory(root_ffhq=path_dataset_ffhq, device=device, mean_latent=mean_latent, max_sample=MAX_SAMPLES)
loaded_model_text = 'without'
if USE_GENE_POOL:
    loaded_model_text = 'with'
    geneFactor.pools = torch.load(path_ckpt_genepool)
print("Gene Pool loaded " + loaded_model_text + " ckpt!")


def tensor2rgb(tensor):
    tensor = (tensor * 0.5 + 0.5) * 255
    tensor = torch.clip(tensor, 0, 255).squeeze(0)
    tensor = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    tensor = tensor.astype(np.uint8)
    return tensor


# Function to optimize the latent vector for editing
def optimize_latent_for_editing(generator, target_image, num_steps=100, lr=0.01):
    # Initialize a random latent vector
    latent_vector = torch.randn(1, 512, device='cuda', requires_grad=True)

    # Optimization loop
    for step in range(num_steps):
        # Generate an image from the current latent vector
        generated_image, _ = generator([latent_vector], return_latents=True, input_is_latent=True)

        # Calculate the loss as the L2 distance between the generated and target images
        loss = torch.nn.functional.mse_loss(generated_image, target_image)

        # Backpropagation and optimization step
        loss.backward()
        latent_vector.data -= lr * latent_vector.grad
        latent_vector.grad.zero_()

    return latent_vector.detach()


# Function to perform random edits on the optimized latent vector
def perform_random_edits(generator, original_latent_vector, num_edits=10, edit_scale=0.1):
    edited_latents = []

    for _ in range(num_edits):
        # Clone the original latent vector to avoid modifying it directly
        latent_vector = original_latent_vector.clone().detach().requires_grad_(True)

        # Apply random perturbations to the latent vector
        perturbations = edit_scale * torch.randn_like(latent_vector)
        latent_vector.data += perturbations

        # Generate an image from the edited latent vector
        edited_image, _ = generator([latent_vector], return_latents=True, input_is_latent=True)

        # Save or display the edited image
        edited_image_np = (edited_image.squeeze().cpu().detach().numpy() + 1) / 2.0
        Image.fromarray((edited_image_np * 255).astype('uint8'))

        edited_latents.append(latent_vector.detach())

    return edited_latents


def generate_child(w18_F, w18_M, random_fakes, gamma=0.46, eta=0.4, dim_to_edit= 10, increase_value=0.5, power_of_dad=0.5):
    w18_syn = fuse_latent(w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
                          random_fakes=random_fakes, fixed_gamma=gamma, fixed_eta=eta, power_of_dad=power_of_dad)

    img_C, generated_latents = generator([w18_syn], return_latents=True, input_is_latent=True)
    
    # GAN inversion part
    # Randomly edit the semantic facial features
    edited_latents = generated_latents.clone()

    # Example: Modify the first 10 dimensions of the latent vector
    # You need to adjust this based on your specific model's architecture and latent vector structure
    edited_latents[0, :dim_to_edit] += increase_value  # Increase the values

    # Generate images from edited latent vectors
    with torch.no_grad():
        edited_image, _ = generator([edited_latents], return_latents=True, input_is_latent=True)

    return edited_image, w18_syn


def synthesize_descendant(pF, pM, attributes=None):
    gender_all = ['male', 'female']
    ages_all = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    if attributes is None:
        attributes = {'age': ages_all[0], 'gender': gender_all[0], 'gamma': 0.47, 'eta': 0.4,
                      'dim_to_edit': 0, 'increase_value': 0.5, 'ethnicity': 'Automatic', 'power_of_dad': 0.5}
    # imgF = align_face(pF)
    # imgM = align_face(pM)
    imgF = load_img(pF)
    imgM = load_img(pM)
    imgF, imgM = imgF.to(device), imgM.to(device)

    father_race = None
    mother_race = None
    if attributes['ethnicity'] == 'Automatic':
        father_race, _, _, _ = predict_race(model_fair_7, imgF.clone(), imgF.device)
        mother_race, _, _, _ = predict_race(model_fair_7, imgM.clone(), imgM.device)
    else:
        father_race = attributes['ethnicity']
        mother_race = attributes['ethnicity']

    w18_1 = encoder(F.interpolate(imgF, size=(256, 256))) + mean_latent
    w18_2 = encoder(F.interpolate(imgM, size=(256, 256))) + mean_latent

    random_fakes = []
    for r in list({father_race, mother_race}):  # search RFGs from Gene Pool
        random_fakes = random_fakes + geneFactor(encoder, w2sub34, attributes['age'], attributes['gender'], r)
    img_C, w18_syn = generate_child(w18_1.clone(), w18_2.clone(), random_fakes,
                                    gamma=attributes['gamma'], eta=attributes['eta'], 
                                    dim_to_edit=attributes['dim_to_edit'], increase_value=attributes['increase_value'], 
                                    power_of_dad=attributes['power_of_dad'])
    
    img_C = tensor2rgb(img_C)
    img_F = tensor2rgb(imgF)
    img_M = tensor2rgb(imgM)

    return img_F, img_M, img_C
