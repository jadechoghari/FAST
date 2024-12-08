import torch
import cv2
import numpy as np
from models.fast import FAST
from mmcv import Config


def load_model(config_path, checkpoint_path, device='cuda'):
    """Load FAST model from config and checkpoint"""
    cfg = Config.fromfile(config_path)
    model = FAST(
        backbone=cfg.model['backbone'],
        neck=cfg.model['neck'],
        detection_head=cfg.model['detection_head']
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, cfg


def preprocess_image(image_path, short_size=736):
    """Preprocess image for inference"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = short_size / min(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    
    # Normalize
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = (img - mean[None, None, :]) / std[None, None, :]
    
    # To tensor
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    
    return img, (h, w)


def run_inference(image_path, config_path, checkpoint_path, device='cuda'):
    """Run inference on single image"""
    # Load model
    model, cfg = load_model(config_path, checkpoint_path, device)
    
    # Preprocess image
    img, org_size = preprocess_image(image_path, cfg.data.test.short_size)
    img = img.to(device)
    
    # Prepare meta info
    img_meta = {
        'org_img_size': [org_size],
        'img_size': [img.shape[2:]]
    }
    
    # Run inference
    with torch.no_grad():
        outputs = model.forward(img, img_metas=img_meta, cfg=cfg.test_cfg)
    
    return outputs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FAST Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on')
    args = parser.parse_args()

    outputs = run_inference(args.image, args.config, args.checkpoint, args.device)
    
    # Print detection outputs
    print("Detection outputs:")
    print("Kernel predictions shape:", outputs['kernels'].shape)
    if 'results' in outputs:
        for i, result in enumerate(outputs['results']):
            print(f"Detection {i}:")
            print("Bboxes shape:", result['bboxes'].shape)
            print("Scores shape:", result['scores'].shape) 