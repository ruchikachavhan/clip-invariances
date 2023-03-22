import torch
import clip
from PIL import Image
from torchvision.datasets import ImageFolder, CIFAR100
import os
from tqdm import tqdm
from transforms import ManualTransform
import torchvision
import numpy as np
from scipy.spatial.distance import mahalanobis
import json

# Create dictionary mapping labels to class name 
imagenet_classes = {}
with open('imagenet_classes.txt', 'r') as f:
    for line in f.readlines():
        _, label, class_name = line.split(' ')
        imagenet_classes[int(label)] = class_name.strip()

# Create dataset class for ImageNet validation set
# Dataset returns image, label, and text description from a dictionary
class ImageNetDataset(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        # Add your custom logic here
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        # text = imagenet_classes[target+1]
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {imagenet_classes[c]}") for c in imagenet_classes.keys()])
        return image, target, text_inputs
    
    def __len__(self):
        return len(self.samples)


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # test dataset
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)
    
    if args.dataset == 'imagenet':
        clean_dataset = ImageNetDataset(os.path.join(args.data_dir, 'val'), transform=preprocess)
    else:
        clean_dataset = CIFAR100(root='../TestDatasets/CIFAR100/', download=True, train=False, transform=preprocess)
    loader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, shuffle=False, num_workers=0)

    clean_path = os.path.join(args.save_dir, 'clean_logits.pt')
    if os.path.exists(clean_path):
        clean_logits = torch.load(clean_path)
    else:
        clean_logits = get_features(loader, model, device, prefix='Clean')
        torch.save(clean_logits, os.path.join(args.save_dir, 'clean_logits.pt'))

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

    aug_path = os.path.join(args.save_dir, '{}_logits.pt'.format(args.transform))
    if os.path.exists(aug_path):
        augmented_logits = torch.load(aug_path)
    else:
        # Load augmented dataset
        transform = ManualTransform(args.transform, k, norm=CLIP_norm_params, resize=args.resize, crop_size=args.crop_size)
        if args.dataset == 'imagenet':
            transformed_dataset = ImageNetDataset(os.path.join(args.data_dir, 'val'), transform = transform)
        else:
            transformed_dataset = CIFAR100(root='../TestDatasets/CIFAR100/', download=True, train=False, transform=transform)
        augmented_loader = torch.utils.data.DataLoader(
            transformed_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=True,
        )
        augmented_logits = get_features(augmented_loader, model, device, prefix=args.transform)
        torch.save(augmented_logits, aug_path)

    clean_logits = clean_logits.to(torch.float32)
    augmented_logits = augmented_logits.to(torch.float32)

    # Calculate KL divergence
    kl_div = torch.nn.KLDivLoss(reduction='none')
    kl = torch.zeros(k)
    for i in range(args.k):
        kl[i] = kl_div(augmented_logits[:, i, :], clean_logits).mean()     

    kl = kl.mean().item()
    # Save KL divergence
    results = {}
    results['transform'] = args.transform
    results['kl'] = kl
    with open(os.path.join(args.save_dir, 'kl_{}.json'.format(args.transform)), 'w') as f:
        json.dump(results, f)

def get_features(loader, model, device, prefix):
    model.eval()
    logits = []
    for i, data in tqdm(enumerate(loader), desc=prefix, total=len(loader)):
        image, label, text = data
        if prefix != 'Clean':
            image = torch.stack(image).squeeze(1)
        image = image.to(device)
        text = text.to(device).squeeze(0)

        # Logits for the image-text classification task
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu()
        logits.append(probs)

        if i % 100 == 0:
            print(f"Processed {i} batches")
    
    return torch.stack(logits)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='../../imagenet1k/')
    parser.add_argument("--save-dir", type=str, default='features/')
    parser.add_argument("--dataset", type=str, default='imagenet')
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