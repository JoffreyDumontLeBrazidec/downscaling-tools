"""A stand-in for `anemoi-training train` with the same start semantics, for the chain tests.

key=value arguments: system.output.checkpoints.root, training.max_steps, training.run_id, training.fork_run_id,
system.input.warm_start, training.load_weights_only. Checkpoints go to <root>/<run_id>/ every 2 steps plus last.ckpt.
"""
import sys
import uuid
from pathlib import Path

import pytorch_lightning as pl
import torch


class Toy(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.mse_loss(self.lin(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:])
    null = lambda v: None if v in (None, "null", "None", "") else v
    root = Path(kv["system.output.checkpoints.root"])
    run_id, warm = null(kv.get("training.run_id")), null(kv.get("system.input.warm_start"))
    weights_only = kv.get("training.load_weights_only", "False") == "True"
    model, ckpt_path = Toy(), None
    if run_id and not weights_only:
        ckpt_path = root / run_id / "last.ckpt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Could not find last checkpoint: {ckpt_path}")
    elif run_id and weights_only:  # the trap: weights load, step and optimiser restart at 0
        model.load_state_dict(torch.load(root / run_id / "last.ckpt", weights_only=False)["state_dict"])
    else:
        if warm:
            model.load_state_dict(torch.load(warm, weights_only=False)["state_dict"])
        run_id = uuid.uuid4().hex
    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(64, 4), torch.randn(64, 1)), batch_size=4)
    ck = pl.callbacks.ModelCheckpoint(dirpath=root / run_id, filename="anemoi-by_step-{step:06d}", every_n_train_steps=2,
                                      save_top_k=-1, save_last=True)
    trainer = pl.Trainer(max_steps=int(kv["training.max_steps"]), callbacks=[ck], logger=False, enable_progress_bar=False,
                         enable_model_summary=False, accelerator="cpu", devices=1)
    trainer.fit(model, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
