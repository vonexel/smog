import os, sys, time, json, gc, math, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import linalg
import wandb
import clip
from scipy import stats as sp_stats
sys.path.append(".")
from src.utils.get_model_and_data import get_model_and_data
from src.utils.tensors import collate
from src.utils.misc import load_model_wo_clip
from src.utils.fixseed import fixseed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str, help="Path to checkpoint file (checkpoint_XXXX.pth.tar)")
    p.add_argument("--sample_size", type=int, default=2000,
                   help="Number of validation samples to evaluate (-1 = full dataset)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--project",     default="smog")
    p.add_argument("--run_name",    default=None)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_runs", type=int, default=3, help="Number of evaluation runs with different seeds")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def l2_normalize(t: torch.Tensor) -> torch.Tensor:
    """
    Apply L2 normalization to input tensor along last dimension
    """
    return torch.nn.functional.normalize(t, p=2, dim=-1)

@torch.no_grad()
def extract_features(model, loader, device, clip_dim):
    """
    Extract paired text and motion features from validation dataset.
    
    Process:
    1. Encode text descriptions using CLIP's text encoder
    2. Encode motion data using MotionCLIP's motion encoder
    3. Normalize both feature sets to unit vectors (L2 norm)
    
    Returns:
      text_feats  : (N, D) - CLIP-encoded text features
      motion_feats: (N, D) - MotionCLIP-encoded motion features
    """
    text_all, motion_all = [], []
    for batch in tqdm(loader, desc="feature-extraction"):
        # Text-to-CLIP encoding
        texts = batch["clip_text"]
        tokenized = clip.tokenize(texts, truncate=True).to(device)
        txt_feat = model.clip_model.encode_text(tokenized).float()   # (B,512)
        txt_feat = l2_normalize(txt_feat)

         # Motion-to-CLIP projection via motion encoder
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        enc_out   = model.encoder(batch_dev)
        mu        = enc_out["mu"]                    # (B, latent_dim)
        if mu.size(1) != clip_dim:
            # Project motion features to CLIP space if dimensions mismatch
            if not hasattr(model, "_eval_proj"):
                model._eval_proj = torch.nn.Linear(mu.size(1), clip_dim, bias=False).to(device)
                torch.nn.init.orthogonal_(model._eval_proj.weight)
                model._eval_proj.eval()
            mu = model._eval_proj(mu)
        motion_feat = l2_normalize(mu)

        text_all.append(txt_feat.cpu())
        motion_all.append(motion_feat.cpu())

    return torch.cat(text_all), torch.cat(motion_all)  # N×D

def r_precision(text_f, motion_f, k_list=(1,5)):
    """
    Calculate R-Precision: Measures cross-modal retrieval accuracy.
    
    Metric Definition:
    - Compute cosine similarity between motion and text features
    - For each query, check if the correct match appears in top-k results
    - Reports mean average precision at k=1 and k=5
    
    Formula:
    R@k = (Number of queries where ground-truth is in top-k matches) / Total queries
    """
    sims = motion_f @ text_f.T                       # (N,N)
    ranks = torch.argsort(sims, dim=1, descending=True)
    results = {}
    for k in k_list:
        #match = (ranks[:, :k] == torch.arange(len(text_f)).unsqueeze(1)).any(dim=1)
        match = (ranks[:, :k] == torch.arange(len(text_f), device=text_f.device).unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = match.float().mean().item()
    return results

def kernel_mmd(x, y, sigma=0.07):
    """
    Compute Maximum Mean Discrepancy (MMD) with Gaussian kernel.
    
    Purpose: Test if two samples come from the same distribution.
    Interpretation:
    - MMD ≈ 0: Similar distributions
    - MMD >> 0: Significant distribution differences
    
    Method:
    1. Compute pairwise distances between samples
    2. Apply Gaussian kernel to distances
    3. Calculate MMD statistic using kernel matrix means
    """
    N, _ = x.shape
    xx = torch.cdist(x, x, p=2.0)**2
    yy = torch.cdist(y, y, p=2.0)**2
    xy = torch.cdist(x, y, p=2.0)**2
    k_xx = torch.exp(-xx / (2*sigma**2)).mean()
    k_yy = torch.exp(-yy / (2*sigma**2)).mean()
    k_xy = torch.exp(-xy / (2*sigma**2)).mean()
    return (k_xx + k_yy - 2*k_xy).item()

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet Distance between two multivariate Gaussians.
    
    Used in Fréchet CLIP Distance (FCD) to compare feature distributions:
    FCD = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2(Σ1Σ2)^½)
    
    Args:
      mu1, mu2: Mean vectors
      sigma1, sigma2: Covariance matrices
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    # Handle numerical stability issues (float)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return (diff@diff + np.trace(sigma1 + sigma2 - 2*covmean)).item()

def compute_fcd(x, y):
    """
    Compute Fréchet CLIP Distance between two sets of features.
    
    Extension of Fréchet Inception Distance (FID) for CLIP features:
    1. Calculate mean/covariance of text and motion features
    2. Compute Frechet distance between distributions
    3. Lower values indicate better alignment between modalities
    """
    x_np, y_np = x.numpy(), y.numpy()
    mu1,  mu2  = x_np.mean(0), y_np.mean(0)
    sigma1, sigma2 = np.cov(x_np, rowvar=False), np.cov(y_np, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def main():
    args = parse_args()
    fixseed(args.seed)
    # Setup paths and configuration
    ckpt_path = Path(args.checkpoint).resolve()
    folder    = ckpt_path.parent
    opt_path  = folder / "opt.yaml"
    assert opt_path.exists(), f"opt.yaml not found in {folder}"

    # Load training configuration from YAML
    parameters = {}
    import yaml, inspect
    with open(opt_path) as fr:
        parameters = yaml.safe_load(fr)
    parameters["device"] = args.device
    # Initialize model and validation dataset
    model, datasets = get_model_and_data(parameters, split="vald")

    val_ds = datasets["test"]
    if args.sample_size > 0 and args.sample_size < len(val_ds):
        idx = np.random.choice(len(val_ds), args.sample_size, replace=False)
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, idx)
    
    # Create data loader with custom collation
    loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=collate)

    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location=args.device)
    load_model_wo_clip(model, state_dict)
    model.eval()

    # Get CLIP feature dimension
    clip_dim = model.clip_model.text_projection.size(1)  # 512
    run = wandb.init(project=args.project,
                     name=args.run_name or f"eval-{ckpt_path.stem}",
                     config={"checkpoint": str(ckpt_path),
                             "sample_size": len(val_ds),
                             "batch_size": args.batch_size,
                             "n_runs": args.n_runs,
                             "clip_dim": clip_dim})

    aggregated = {m: [] for m in ["R@1", "R@5", "MMD", "FCD"]}
    time_start = time.time()
    for r in range(args.n_runs):
        print(f"\nRun {r+1}/{args.n_runs}")
        fixseed(args.seed + r*100)

        text_f, motion_f = extract_features(model, loader, args.device, clip_dim)
        text_f = text_f.to(args.device)
        motion_f = motion_f.to(args.device)

        r_dict = r_precision(text_f, motion_f, k_list=(1,5))
        mmd    = kernel_mmd(text_f, motion_f)
        fcd    = compute_fcd(text_f.cpu(), motion_f.cpu())

        # Wandb logging
        wandb.log({**r_dict, "MMD": mmd, "FCD": fcd, "run": r}, step=r)
        for k,v in r_dict.items(): aggregated[k].append(v)
        aggregated["MMD"].append(mmd)
        aggregated["FCD"].append(fcd)

        del text_f, motion_f
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Final statistics and confidence intervals
    summary = {}
    for k, vals in aggregated.items():
        vals = np.array(vals)
        summary[f"{k}_mean"] = vals.mean()
        summary[f"{k}_std"]  = vals.std()
        if len(vals) > 1:
            ci_low, ci_hi = sp_stats.t.interval(0.95, len(vals)-1,
                                                loc=vals.mean(),
                                                scale=sp_stats.sem(vals))
            summary[f"{k}_ci95_low"]  = ci_low
            summary[f"{k}_ci95_high"] = ci_hi
        else:  
            summary[f"{k}_ci95_low"] = summary[f"{k}_ci95_high"] = vals.mean()

    wandb.summary.update(summary)
    wandb.finish()

    # Save results to json
    out_dir = folder / "evaluation_results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir/"results.json", "w") as fw:
        json.dump({"runs": aggregated, "summary": summary,
                   "config": vars(args)}, fw, indent=2)

    print("\n=== Evaluation completed ===")
    for k in ["R@1", "R@5", "MMD", "FCD"]:
        print(f"{k}: {summary[k+'_mean']:.4f}  ±{summary[k+'_std']:.4f}")
    print(f"Total time: {time.time()-time_start:.1f}s")

if __name__ == "__main__":
    main()
