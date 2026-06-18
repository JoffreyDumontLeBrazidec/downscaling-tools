import torch, sys
sys.path.insert(0, "/home/ecm5702/dev/downscaling-tools")
from interp.model_utils import load_model, prepare_batch

bundle = load_model("/home/ecm5702/scratch/aifs/checkpoint/85884ee70d8749609087d205c0b15605/anemoi-by_epoch-epoch_144-step_189950.ckpt", device="cuda", precision="fp32")

bundle.datamodule.setup(stage="predict")
dl = bundle.datamodule.val_dataloader()

# Get 2 batches
batches = []
for i, b in enumerate(dl):
    batches.append(b)
    if len(batches) >= 2:
        break

x_lres_all = torch.cat([b[0] for b in batches], dim=0)
x_hres_all = torch.cat([b[1] for b in batches], dim=0)
y_all = torch.cat([b[2] for b in batches], dim=0)
print("Stacked shapes (before prepare):", x_lres_all.shape, x_hres_all.shape, y_all.shape)

# Prepare batch with detailed logging
prepared = prepare_batch(bundle, x_lres_all, x_hres_all, y_all)
print("x_interp:", prepared["x_interp"].shape)
print("x_hres:", prepared["x_hres"].shape)
print("y_residual:", prepared["y_residual"].shape)
print("x_interp_raw:", prepared["x_interp_raw"].shape)

# Now test denoise_at_sigma
sigma = 0.1
noise = torch.randn_like(prepared["y_residual"])
print("\nPreparing for denoise_at_sigma...")
print("noise shape:", noise.shape)

inner = bundle.inner_model
batch_size = prepared["x_interp"].shape[0]
ensemble_size = prepared["x_interp"].shape[2]
sigma_t = torch.tensor(sigma, device="cuda", dtype=prepared["x_interp"].dtype)
sigma_4d = sigma_t.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1)
y_noised = prepared["y_residual"] + sigma_t * noise
print("sigma_4d:", sigma_4d.shape)
print("y_noised:", y_noised.shape)

# Manually trace _assemble_input
x_interp = prepared["x_interp"]
x_hres = prepared["x_hres"]
print("\nBefore _assemble_input:")
print("  x_interp:", x_interp.shape, "dtype:", x_interp.dtype)
print("  x_hres:", x_hres.shape, "dtype:", x_hres.dtype)
print("  y_noised:", y_noised.shape, "dtype:", y_noised.dtype)

import einops
bse = batch_size * ensemble_size
x_interp_2d = einops.rearrange(x_interp, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")
x_hres_2d = einops.rearrange(x_hres, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")
y_noised_2d = einops.rearrange(y_noised, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")
print("\nAfter rearrange to 2D:")
print("  x_interp_2d:", x_interp_2d.shape)
print("  x_hres_2d:", x_hres_2d.shape)
print("  y_noised_2d:", y_noised_2d.shape)

node_attr = inner.node_attributes("data", batch_size=bse)
print("  node_attr:", node_attr.shape)

total_feat = x_interp_2d.shape[-1] + x_hres_2d.shape[-1] + y_noised_2d.shape[-1] + node_attr.shape[-1]
print("\nTotal concat features:", total_feat, "= ", x_interp_2d.shape[-1], "+", x_hres_2d.shape[-1], "+", y_noised_2d.shape[-1], "+", node_attr.shape[-1])
print("Model expects: 159")
