import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy import ndimage
from segment_anything import sam_model_registry


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class SAMFinetune(nn.Module):
    def __init__(self, model_type='vit_b', checkpoint=None):
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.learned_sparse_embedding = nn.Parameter(torch.randn(1, 1, self.sam.prompt_encoder.embed_dim) * 0.02)
        self.no_mask_embed = self.sam.prompt_encoder.no_mask_embed
        for p in self.sam.image_encoder.parameters(): p.requires_grad = False
        for p in self.sam.prompt_encoder.parameters(): p.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(images)
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        dense_emb = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, *self.sam.prompt_encoder.image_embedding_size)
        masks, ious = [], []
        for i in range(images.shape[0]):
            m, iou = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i:i+1], image_pe=image_pe,
                sparse_prompt_embeddings=self.learned_sparse_embedding,
                dense_prompt_embeddings=dense_emb, multimask_output=False)
            masks.append(m)
            ious.append(iou)
        return torch.cat(masks), torch.cat(ious)


_sam_model = None
_device = None


def load_sam(finetune_checkpoint, device=None, base_checkpoint=None):
    global _sam_model, _device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _device = device
    _sam_model = SAMFinetune(checkpoint=base_checkpoint).to(device)
    ckpt = torch.load(finetune_checkpoint, map_location=device, weights_only=False)
    _sam_model.load_state_dict(ckpt['model_state_dict'])
    _sam_model.eval()
    print("SAM loaded")


def free_sam():
    """Free SAM model from GPU memory."""
    global _sam_model
    if _sam_model is not None:
        del _sam_model
        _sam_model = None
        torch.cuda.empty_cache()
        print("SAM freed")


def postprocess_mask(mask, threshold=0.5, morph_kernel_size=5, smooth_kernel_size=5, fill_holes=True):
    binary = (mask > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        binary = (labels == largest_label).astype(np.uint8)
    if fill_holes:
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
    if smooth_kernel_size > 0:
        smoothed = cv2.GaussianBlur(binary.astype(np.float32), (smooth_kernel_size, smooth_kernel_size), 0)
        binary = (smoothed > 0.5).astype(np.uint8)
    return binary


def segment_image(img_rgb):
    h, w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (1024, 1024))
    img_norm = (img_resized / 255.0 - MEAN) / STD
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(_device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        pred_mask, _ = _sam_model(img_tensor)
        mask = torch.sigmoid(pred_mask).float().cpu().numpy()[0, 0]
    mask = cv2.resize(mask, (w, h))
    return postprocess_mask(mask)


def segment_images_batch(images_rgb, batch_size=4):
    """Segment multiple images with batched encoder inference.

    Args:
        images_rgb: list of RGB numpy arrays (can be different sizes)
        batch_size: number of images per encoder forward pass

    Returns:
        list of binary mask arrays (same sizes as inputs)
    """
    # Preprocess all images
    orig_sizes = []
    tensors = []
    for img in images_rgb:
        h, w = img.shape[:2]
        orig_sizes.append((h, w))
        img_resized = cv2.resize(img, (1024, 1024))
        img_norm = (img_resized / 255.0 - MEAN) / STD
        tensors.append(torch.from_numpy(img_norm.transpose(2, 0, 1)).float())

    masks_out = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start:start + batch_size]).to(_device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred_masks, _ = _sam_model(batch)
            pred_masks = torch.sigmoid(pred_masks).float().cpu().numpy()
        for k in range(pred_masks.shape[0]):
            idx = start + k
            h, w = orig_sizes[idx]
            mask = cv2.resize(pred_masks[k, 0], (w, h))
            masks_out.append(postprocess_mask(mask))

    return masks_out
