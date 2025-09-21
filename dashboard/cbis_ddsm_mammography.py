# cbis_ddsm_mammography.py
import cv2, torch, timm, numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
CBIS-DDSM Mammography Pipeline

Author: (Me)

What this module does:
  • Defines my image normalization / transforms (CLAHE + Resize + Normalize).
  • Implements a MaxViT-Tiny classifier head that fuses image features with simple metadata.
  • Provides Grad-CAM++ and a lightweight gradient-based SHAP-style attribution.
  • Exposes a CBISDDSMPipeline that ties together:
        - YOLO detection (get the highest-confidence lesion crop),
        - MaxViT classification (benign vs malignant),
        - Visual explanations (Grad-CAM++ & gradient SHAP overlays).

Design notes:
  • I keep the transforms explicit and CPU-friendly; model tensors live on the chosen device.
  • Metadata is a 4-dim one-hot vector: [LEFT, RIGHT, CC, MLO].
  • All visualization utilities return uint8 RGB overlays suitable for Streamlit display.
"""

# ---- constants / transforms
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Validation/Inference transform:
#   - CLAHE to enhance local contrast in mammograms.
#   - Resize to model input.
#   - Normalize by ImageNet stats (consistent with MaxViT pretraining conventions).
#   - Convert to torch tensor.
val_tf = A.Compose([
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8,8), p=1.0),
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ToTensorV2(),
])

# ----------------- Models -----------------
class MaxViTTinyWithMeta(nn.Module):
    """
    MaxViT-Tiny backbone (timm) + small MLP head that fuses image features with metadata.

    Args:
      meta_dim  : dimension of my metadata vector [L, R, CC, MLO] (default 4).
      dropout   : dropout before the final linear layer.
      prior_bias: bias init for the final logit (log-odds prior); 0 means 50/50 prior.

    Forward:
      x    : image tensor (B, 3, H, W) after my val_tf transform.
      meta : metadata tensor (B, meta_dim).
      Returns: raw logit (B, 1). I apply sigmoid at call sites as needed.
    """
    def __init__(self, meta_dim=4, dropout=0.4, prior_bias=0.0):
        super().__init__()
        # I use the TF MaxViT Tiny variant from timm, with global average pooling and no head.
        self.base = timm.create_model(
            "maxvit_tiny_tf_224", pretrained=False, in_chans=3, num_classes=0, global_pool="avg"
        )
        feat_dim = self.base.num_features
        self.meta_norm = nn.Identity()  # placeholder if I want to add BatchNorm/LayerNorm to meta
        self.head = nn.Sequential(
            nn.Linear(feat_dim + meta_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        # Initialize final bias to a prior if I want to skew the operating point early in training.
        with torch.no_grad():
            self.head[-1].bias.fill_(prior_bias)

    def forward(self, x, meta):
        feats = self.base(x)                          # (B, feat_dim)
        fused = torch.cat([feats, self.meta_norm(meta)], dim=1)
        return self.head(fused)                       # (B, 1) raw logit


# ----------------- Grad-CAM++ -----------------
def get_fixed_cam_layer(backbone: nn.Module) -> nn.Module:
    """
    Heuristic: pick a deep Conv2d from the last stage for Grad-CAM++ hooks.
    I attempt a specific path (stages[-1].blocks[-1]) and fall back to the last Conv2d found.
    """
    last_conv = None
    try:
        last_stage = backbone.stages[-1]
        last_block = last_stage.blocks[-1]
        for m in last_block.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
    except Exception:
        pass
    if last_conv is None:
        for m in backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
    return last_conv


class GradCAMPP:
    """
    Grad-CAM++ explanation:
      • Hooks activations and gradients on a chosen conv layer.
      • Uses the sign-adjusted target (mode="pred") to explain the predicted class.

    Usage:
      gradcam = GradCAMPP(model, target_layer)
      heatmap, probs = gradcam(x, meta, mode="pred"|"malignant"|"benign")
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.layer = target_layer
        self.activations = None
        self.gradients = None
        # Register hooks to capture forward activations and backward grads.
        self.h_act = self.layer.register_forward_hook(self._save_act)
        self.h_grad = self.layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, inp, out):  self.activations = out
    def _save_grad(self, m, gin, gout): self.gradients = gout[0]

    def __call__(self, x, meta, mode="pred"):
        # Forward pass to get logits and probabilities
        logits = self.model(x, meta)               # (B,1)
        probs  = torch.sigmoid(logits).squeeze(1)

        # Choose which class to attribute:
        if mode == "malignant":
            target = logits.squeeze(1)
        elif mode == "benign":
            target = -logits.squeeze(1)
        else:
            # "pred" mode: sign(logit) * logit, which attributes toward the predicted side.
            sign = torch.where(logits.squeeze(1) >= 0, 1.0, -1.0)
            target = sign * logits.squeeze(1)

        # Backward to get gradients at the target conv layer
        self.model.zero_grad(set_to_none=True)
        target.sum().backward(retain_graph=True)

        # Pull cached activations (A) and gradients (dY)
        A   = self.activations                    # (B,C,H,W)
        dY  = self.gradients                      # (B,C,H,W)

        # Grad-CAM++ weighting
        dY2 = dY.pow(2)
        dY3 = dY.pow(3)
        sumA = A.sum(dim=(2,3), keepdim=True)
        eps = 1e-8
        alpha = dY2 / (2.0 * dY2 + sumA * dY3 + eps)
        weights = (alpha * torch.relu(dY)).sum(dim=(2,3))      # (B,C)

        # Weighted sum over channels -> CAM
        cam = (weights[:, :, None, None] * A).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = torch.relu(cam)

        # Normalize each CAM to [0,1]
        cam_min = cam.amin(dim=(1, 2, 3), keepdim=True)
        cam_max = cam.amax(dim=(1, 2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze(1), probs.detach()  # (B,H,W), (B,)


# ----------------- SHAP (gradient variant) -----------------
def gradient_shap(model, x, meta, n_samples=12, stdev=0.2):
    """
    Extremely lightweight gradient-based attribution that samples noisy interpolations
    between a zero baseline and the input, accumulating gradients as feature importance.

    Args:
      model     : classifier taking (x, meta)
      x         : input image tensor (B, 3, H, W); I typically pass a single-sample batch.
      meta      : metadata tensor (B, 4)
      n_samples : number of noisy samples (trade-off between speed and smoothness)
      stdev     : Gaussian noise std used in sampling

    Returns:
      (B, H, W) heatmap (float32 in [0,1]) highlighting salient pixels.
    """
    model.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    baseline = torch.zeros_like(x)
    attr = torch.zeros_like(x)
    for _ in range(n_samples):
        eps = torch.randn_like(x) * stdev
        alpha = torch.rand(1, device=x.device)
        x_interp = baseline + alpha * (x - baseline) + eps
        prob = torch.sigmoid(model(x_interp, meta)).sum()
        grad = torch.autograd.grad(prob, x, retain_graph=False, create_graph=False)[0]
        attr += grad
    attr /= float(n_samples)
    # Channel-wise magnitude → normalize to [0,1]
    m = attr.abs().sum(dim=1, keepdim=True)
    m = (m - m.min())/(m.max()-m.min()+1e-8)
    return m.squeeze(1).detach()


# ----------------- Utility -----------------
def overlay_heatmap(rgb_uint8, heat_01, blur_sigma=0.6, gamma=0.85):
    """
    Overlay a single-channel heatmap (0..1) onto an RGB uint8 image using JET colormap.

    Args:
      rgb_uint8 : base RGB image (H, W, 3) in uint8
      heat_01   : heatmap tensor/array in [0,1]
      blur_sigma: optional Gaussian blur to smooth attributions
      gamma     : gamma correction to accentuate mid-high activations

    Returns:
      RGB uint8 image with heatmap overlay.
    """
    heat_np = heat_01.detach().cpu().numpy()
    heat_np = np.clip(heat_np, 0, 1) ** gamma
    hm = (heat_np * 255).astype(np.uint8)
    if blur_sigma and blur_sigma > 0:
        hm = cv2.GaussianBlur(hm, (0, 0), blur_sigma)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(rgb_uint8, 0.55, hm, 0.45, 0)

def to_3ch_gray(rgb_uint8):
    """
    Convert any RGB uint8 image to 3-channel grayscale (replicated) as uint8.
    My classifier expects 3 channels even for grayscale mammograms.
    """
    g = np.array(Image.fromarray(rgb_uint8).convert("L"))
    return np.stack([g, g, g], axis=2).astype(np.uint8)


# ----------------- Pipeline -----------------
class CBISDDSMPipeline:
    """
    End-to-end mammography pipeline on CBIS-DDSM-style inputs.

    Steps:
      1) detect(): run YOLO on the full mammogram → keep top-1 box and return the crop.
      2) classify(): run MaxViT-Tiny-with-meta on the crop → benign/malignant + confidence.
      3) explain(): visualize Grad-CAM++ and gradient SHAP overlays for the last classification.

    Args:
      yolo_path : path to YOLO weights (.pt)
      clf_path  : path to my MaxViT state_dict (strict=True load)
      device    : torch.device or None (auto CUDA if available)
      thresh    : operating threshold for "malignant" (sigmoid ≥ thresh)

    Note:
      set_metadata() MUST be called with laterality & view before classify()/explain().
    """
    def __init__(self, yolo_path: str, clf_path: str, device=None, thresh=0.5):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thresh = thresh

        # Metadata: [L, R, CC, MLO] one-hot; default zeros until caller sets it.
        self.meta = torch.zeros(1,4, device=self.device)  # [L,R,CC,MLO]

        # --- load models ---
        self.detector = YOLO(yolo_path)
        self.clf = MaxViTTinyWithMeta(prior_bias=0.0).to(self.device)

        # Load state dict. I accept either a plain state_dict or a dict with "model".
        state = torch.load(clf_path, map_location=self.device)
        state = state["model"] if (isinstance(state, dict) and "model" in state) else state
        self.clf.load_state_dict(state, strict=True)
        self.clf.eval()

        # Prepare Grad-CAM++ with a sensible last conv layer
        self.gradcam = GradCAMPP(self.clf, get_fixed_cam_layer(self.clf.base))

    def set_metadata(self, laterality: str, view: str):
        """
        Set my metadata vector from textual laterality/view:
          laterality ∈ {"LEFT","RIGHT"}, view ∈ {"CC","MLO"} (case-insensitive).
        """
        L = 1.0 if laterality.upper()=="LEFT"  else 0.0
        R = 1.0 if laterality.upper()=="RIGHT" else 0.0
        CC= 1.0 if view.upper()=="CC"         else 0.0
        MLO=1.0 if view.upper()=="MLO"        else 0.0
        self.meta = torch.tensor([[L,R,CC,MLO]], dtype=torch.float32, device=self.device)

    def detect(self, full_rgb_uint8, conf=0.25, iou=0.45):
        """
        Run YOLO on the full image and return:
          - annotated full image (RGB),
          - best crop (RGB) for the top-1 confidence box (or None if no box),
          - bbox tuple (x1,y1,x2,y2) in the full image coordinates.

        I convert to BGR for YOLO, keep imgsz=1024 as a good trade-off.
        """
        img_bgr = cv2.cvtColor(full_rgb_uint8, cv2.COLOR_RGB2BGR)
        res = self.detector.predict(source=img_bgr, imgsz=1024, conf=conf, iou=iou, verbose=False)[0]
        full_annot_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)

        # No detections → return annotated full image and Nones
        if (res.boxes is None) or (len(res.boxes)==0):
            return full_annot_rgb, None, None

        # Choose the highest-confidence detection
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes[idx].astype(int)

        # Clip to valid bounds
        h, w = img_bgr.shape[:2]
        x1, y1 = max(0,x1), max(0,y1); x2, y2 = min(w-1,x2), min(h-1,y2)

        # Extract the crop in RGB
        crop_rgb = full_rgb_uint8[y1:y2, x1:x2].copy()
        bbox = (x1, y1, x2, y2)
        return full_annot_rgb, crop_rgb, bbox

    def _preprocess_tensor(self, crop_rgb_uint8):
        """
        Convert the RGB crop to 3-channel grayscale, apply val_tf, return a 4D tensor (1,3,H,W).
        """
        x3 = to_3ch_gray(crop_rgb_uint8)
        t  = val_tf(image=x3)["image"].unsqueeze(0).to(self.device)
        return t

    def classify(self, crop_rgb_uint8):
        """
        Classify a lesion crop as benign/malignant using my operating threshold.

        Returns:
          label : "malignant" | "benign"
          conf  : confidence w.r.t. the predicted label (p or 1-p)
          p_mal : raw malignant probability (sigmoid(logit))
          x     : preprocessed tensor used for this prediction (for explanations)
        """
        x = self._preprocess_tensor(crop_rgb_uint8)
        with torch.no_grad():
            logit = self.clf(x, self.meta)
            p_mal = torch.sigmoid(logit).item()
        is_mal = p_mal >= self.thresh
        label  = "malignant" if is_mal else "benign"
        conf   = p_mal if is_mal else (1.0 - p_mal)
        return label, conf, p_mal, x  # return x for explanations

    def explain(self, x_tensor):
        """
        Produce Grad-CAM++ and gradient-SHAP overlays for the last classified tensor.

        Returns:
          cam_img  : RGB uint8 overlay of Grad-CAM++
          shap_img : RGB uint8 overlay of gradient-based SHAP approximation
        """
        cam_batch, _ = self.gradcam(x_tensor, self.meta, mode="pred")
        cam  = cam_batch[0]  # (H,W)
        shap = gradient_shap(self.clf, x_tensor, self.meta)[0]

        # Reconstruct a uint8 RGB view from the normalized tensor (inverse of Normalize)
        base_rgb = (x_tensor[0].permute(1,2,0).detach().cpu().numpy()*IMAGENET_STD + IMAGENET_MEAN)
        base_rgb = np.clip(base_rgb*255.0, 0, 255).astype(np.uint8)

        cam_img  = overlay_heatmap(base_rgb, cam)
        shap_img = overlay_heatmap(base_rgb, shap)
        return cam_img, shap_img
