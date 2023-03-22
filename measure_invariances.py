import torch
import clip
from PIL import Image
from torchvision.datasets import ImageFolder
import os
from tqdm import tqdm
from transforms import ManualTransform
import torchvision
import numpy as np
from scipy.spatial.distance import mahalanobis
import json



def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)
    print("Preprocessing", preprocess)

    clean_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), transform = preprocess)
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True,
    )


    clean_path = os.path.join(args.save_dir, 'clean_features.pt')
    if os.path.exists(clean_path):
        clean_features = torch.load(clean_path)
    else:
        clean_features = get_features(clean_loader, model, device, prefix='Clean')
        # Save clean features
        torch.save(clean_features, os.path.join(args.save_dir, 'clean_features.pt'))

    clean_features = clean_features.to(dtype=torch.float32)
    clean_features = clean_features.squeeze(1)
    mean_feature = clean_features.mean(dim=0)
    cov_matrix = np.cov(clean_features, rowvar=False)
    cholesky_matrix = torch.cholesky_inverse(torch.from_numpy(cov_matrix)).to(dtype=torch.float32)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Create a transform with CLIP normalization
    CLIP_norm_params = [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]]

    if args.k is not None:
        k = args.k
    elif args.transform in ['h_flip', 'v_flip', 'grayscale', 'invert', 'equalize']:
        k = 2
    elif args.transform == 'posterize':
        k = 7
    else:
        k = 256
    args.k = k

    aug_path = os.path.join(args.save_dir, '{}_features.pt'.format(args.transform))
    if os.path.exists(aug_path):
        augmented_features = torch.load(aug_path)
    else:
        transform = ManualTransform(args.transform, k, norm=CLIP_norm_params, resize=args.resize, crop_size=args.crop_size)
        transformed_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), transform = transform)
        augmented_loader = torch.utils.data.DataLoader(
            transformed_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=True,
        )
        augmented_features = get_features(augmented_loader, model, device, prefix=args.transform)
        # Save augmented features
        torch.save(augmented_features, os.path.join(args.save_dir, '{}_features.pt'.format(args.transform)))
    
    # Need to convert dtype due to stupid error that cosine similarity doesn't work with float16
    augmented_features = augmented_features.to(dtype=torch.float32)

    
    # Compute cosine similarity
    sim_matrix = torch.zeros((augmented_features.shape[0], args.k))
    for i in range(args.k):
        a = (mean_feature - clean_features).cpu() @ cholesky_matrix
        b = (mean_feature - augmented_features[:, i, :]).cpu() @ cholesky_matrix
        sim_matrix[:, i] = cosine_similarity(a, b)

    # Compute mahalanobis distance
    dist_matrix = torch.zeros((augmented_features.shape[0], args.k))
    for i in range(augmented_features.shape[0]):
        for j in range(args.k):
            dist_matrix[i, j] = mahalanobis(clean_features[i].cpu(), augmented_features[i, j, :].cpu(), inv_cov_matrix)

    dist_matrix = torch.from_numpy(np.nanmean(dist_matrix, axis=0))
    sim_matrix = torch.from_numpy(np.nanmean(sim_matrix, axis=0))
    sim = sim_matrix.mean()
    dist = dist_matrix.mean()


    # Save similarities of image encoder in json file
    results = {}
    results['transform'] = args.transform
    results['cosine'] = sim.item()
    results['mahalanobis'] = dist.item()
    with open(os.path.join(args.save_dir, 'similarities_{}.json'.format(args.transform)), 'w') as f:
        json.dump(results, f)
        

def cosine_similarity(x, y):
    # Manually implement cosine similarity
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    return torch.einsum('...i,...i->...', x, y)

def get_features(loader, model, device, prefix):
    features = []
    for i, (image, _) in tqdm(enumerate(loader), desc=prefix, total=len(loader)):
        if prefix != 'Clean':
            image = torch.stack(image).squeeze(1)
        image = image.to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        features.append(image_features.cpu())
        if i % 100 == 0:
            print(f"Processed {i} batches")

    return torch.stack(features)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='../../imagenet1k/')
    parser.add_argument("--save-dir", type=str, default='features/')
    parser.add_argument("--model", type=str, default='ViT-B/32', help='Type of Model')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--transform", type=str, required=True)
    parser.add_argument('--k', default=None, type=int, help='number of transformations')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--resize', default=256, type=int, metavar='R', help='resize')
    parser.add_argument('--crop-size', default=224, type=int, metavar='C',help='crop size')
    args = parser.parse_args()

    main(args)

