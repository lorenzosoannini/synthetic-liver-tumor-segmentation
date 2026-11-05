from pathlib import Path
import csv
import numpy as np
import nibabel as nib
from scipy import ndimage
import torch
import lightning.pytorch as pl
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import Invertd
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
from upstream_reference.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


def validation_dice(pred, target, native_shape):
    """Match monai_trainer.py: nearest zoom of discrete maps, empty target -> 0."""
    ratios = np.asarray(native_shape, dtype=float) / np.asarray(pred.shape)
    pred = ndimage.zoom(pred, ratios, order=0, prefilter=False)
    target = ndimage.zoom(target, ratios, order=0, prefilter=False)
    scores = []
    for cls in (1, 2):
        p, y = pred == cls, target == cls
        scores.append(2.0 * np.count_nonzero(p & y) / (p.sum() + y.sum()) if y.any() else 0.0)
    return scores


def denoise_prediction(pred):
    """Reference validation.py: largest liver + tumor within 30-voxel dilation."""
    regions, count = ndimage.label(pred == 1)
    liver = np.zeros(pred.shape, dtype=bool)
    if count:
        sizes = np.bincount(regions.ravel())
        sizes[0] = 0
        liver = regions == sizes.argmax()
    tumor = (pred == 2) & ndimage.binary_dilation(liver, iterations=30)
    out = np.zeros(pred.shape, dtype=np.uint8)
    out[liver], out[tumor] = 1, 2
    return out


def native_metrics(pred, target, spacing):
    scores = {}
    for cls, name in ((1, 'liver'), (2, 'tumor')):
        p, y = pred == cls, target == cls
        denominator = int(p.sum() + y.sum())
        # Both empty: undefined, excluded from mean and counted separately.
        scores[f'{name}_dice'] = 2. * np.count_nonzero(p & y) / denominator if denominator else float('nan')
        if not denominator:
            nsd = float('nan')
        elif not p.any() or not y.any():
            nsd = 0.
        else:
            distances = compute_surface_distances(y, p, spacing_mm=spacing)
            nsd = compute_surface_dice_at_tolerance(distances, tolerance_mm=2.)
        scores[f'{name}_nsd'] = float(nsd)
    return scores


class UNet3DLightning(pl.LightningModule):
    def __init__(self, max_epochs=4000, warmup_epochs=100, lr=4e-4,
                 prediction_dir='runs/reference/predictions'):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(spatial_dims=3, in_channels=1, out_channels=3,
                          channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True,
                                  smooth_nr=0, smooth_dr=1e-6)
        self.test_rows = []

    def forward(self, x):
        return self.model(x)

    def infer(self, x):
        return sliding_window_inference(x, roi_size=(96, 96, 96), sw_batch_size=1,
                                        predictor=self.forward, overlap=0.5, mode='gaussian')

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(self(batch['image']), batch['label'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=batch['image'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.infer(batch['image'])
        self.log('val_loss', self.loss_fn(logits, batch['label']), batch_size=1)
        sample = decollate_batch(batch)[0]
        pred = logits[0].argmax(0).detach().cpu().numpy()
        target = sample['label'][0].detach().cpu().numpy()
        scores = validation_dice(pred, target, sample['image'].meta['spatial_shape'])
        self.log('val_liver_dice', scores[0], batch_size=1)
        self.log('val_tumor_dice', scores[1], batch_size=1, prog_bar=True)
        self.log('val_mean_dice', float(np.mean(scores)), batch_size=1, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_rows = []

    def test_step(self, batch, batch_idx):
        logits = self.infer(batch['image'])
        # Invert continuous logits BEFORE argmax, exactly as reference evaluation.
        sample = decollate_batch(batch)[0]
        sample['image'] = sample['image'].cpu()
        sample['pred'] = logits[0].detach().cpu()
        inverse = Invertd(keys='pred', transform=self.trainer.datamodule.test_transform,
                          orig_keys='image', nearest_interp=False, to_tensor=True, device='cpu')
        sample = inverse(sample)
        pred = sample['pred'].argmax(0).numpy().astype(np.uint8)
        label = sample['label'][0].cpu().numpy().astype(np.uint8)
        if pred.shape != label.shape:
            raise ValueError(f'Native grid mismatch: {pred.shape} != {label.shape}')
        affine = sample['label'].affine.cpu().numpy()
        if not np.allclose(sample['pred'].affine.cpu().numpy(), affine, atol=1e-4):
            raise ValueError('Prediction and label native affines differ.')
        pred = denoise_prediction(pred)
        name = Path(sample['label'].meta['filename_or_obj']).name
        scores = native_metrics(pred, label, nib.affines.voxel_sizes(affine))
        self.test_rows.append({'name': name, **scores})
        out = Path(self.hparams.prediction_dir)
        out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred, affine), out / name)

    def on_test_epoch_end(self):
        out = Path(self.hparams.prediction_dir)
        out.mkdir(parents=True, exist_ok=True)
        keys = ('liver_dice', 'liver_nsd', 'tumor_dice', 'tumor_nsd')
        with (out / 'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', *keys])
            writer.writeheader()
            writer.writerows(self.test_rows)
        for key in keys:
            values = np.asarray([r[key] for r in self.test_rows])
            valid = np.isfinite(values)
            self.log(f'test_{key}', float(values[valid].mean()) if valid.any() else float('nan'))
            self.log(f'test_{key}_valid_cases', float(valid.sum()))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}}
