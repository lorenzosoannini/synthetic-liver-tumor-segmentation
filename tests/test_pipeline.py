import json
import random
from pathlib import Path

import nibabel as nib
import nbformat
import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer, LightningDataModule
from monai.data import MetaTensor, CacheDataset, Dataset, DataLoader, list_data_collate
from monai.transforms import Compose

from syntumor.data import (GenerateSyntheticTumor, get_test_transform, spatial_transforms,
                           build_splits, isolated_rng, get_train_transform)
from syntumor.model import UNet3DLightning, denoise_prediction, native_metrics, validation_dice
from upstream_reference.TumorGenerated.utils import SynthesisTumor, get_predefined_texture


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def volume():
    shape = (48, 48, 48)
    image = np.full(shape, 100, np.float32)
    label = np.zeros(shape, np.uint8)
    label[7:41, 7:41, 7:41] = 1
    return image, label


@pytest.mark.parametrize('kind', ['tiny', 'small', 'medium', 'large', 'mix'])
def test_upstream_synthesis_labels_match_intensity(volume, kind):
    image, label = volume
    with isolated_rng(2):
        texture = get_predefined_texture((64, 64, 64), 3, 4)
        out, target = SynthesisTumor(image.copy(), label.copy(), kind, texture)
    assert (target == 2).any()
    assert np.all(label[target == 2] == 1)
    assert np.all(out[target == 2] < image[target == 2])
    np.testing.assert_array_equal(out[label == 0], image[label == 0])
    assert set(np.unique(target)) == {0, 1, 2}


def test_transform_matches_reference_preserves_metadata_and_inputs(volume):
    image, label = volume
    bank = [np.full((64, 64, 64), .5)] * 10
    transform = GenerateSyntheticTumor(bank, prob=1)
    transform.set_random_state(seed=123)
    # Independently replay the wrapper's choices to compare to unchanged upstream.
    rng = np.random.RandomState(123)
    rng.rand()
    idx = rng.randint(10)
    kind = str(rng.choice(transform.tumor_types))
    seed = rng.randint(2**32 - 1)
    with isolated_rng(seed):
        expected_image, expected_label = SynthesisTumor(image.copy(), label.copy(), kind, bank[idx])
    affine = torch.diag(torch.tensor([1., 1., 1., 1.], dtype=torch.float64))
    affine[:3, 3] = torch.tensor([10., -20., 30.])
    data = {'image': MetaTensor(image[None].copy(), affine=affine),
            'label': MetaTensor(label[None].copy(), affine=affine)}
    result = transform(data)
    np.testing.assert_array_equal(result['image'][0], expected_image)
    np.testing.assert_array_equal(result['label'][0], expected_label)
    torch.testing.assert_close(result['image'].affine, affine)
    torch.testing.assert_close(result['label'].affine, affine)
    np.testing.assert_array_equal(data['image'][0], image)
    np.testing.assert_array_equal(data['label'][0], label)
    transform.set_random_state(seed=123)
    repeated = transform(data)
    torch.testing.assert_close(repeated['image'], result['image'])


def test_rng_restored():
    random.seed(7); np.random.seed(7)
    expected = random.random(), np.random.rand()
    random.seed(7); np.random.seed(7)
    with isolated_rng(31):
        random.random(); np.random.rand()
    assert (random.random(), np.random.rand()) == expected


def write_case(tmp_path):
    image = np.full((32, 28, 24), -21, np.float32)
    label = np.zeros(image.shape, np.uint8)
    label[4:28, 4:24, 3:21] = 1
    label[12:20, 10:18, 8:16] = 2
    image[label == 1] = 80
    image[label == 2] = 150
    affine = np.diag([-1.5, 2., 3., 1.])
    affine[:3, 3] = [30., -10., 7.]
    record = {}
    for key, array in [('image', image), ('label', label)]:
        path = tmp_path / f'{key}.nii.gz'
        nib.save(nib.Nifti1Image(array, affine), path)
        record[key] = str(path)
    return record, affine, label


def test_spacing_before_synthesis_and_cache_stays_healthy(tmp_path):
    record, affine, label = write_case(tmp_path)
    label[label == 2] = 1
    nib.save(nib.Nifti1Image(label, affine), record['label'])
    bank = [np.full((128, 128, 128), .5)] * 10
    synth = GenerateSyntheticTumor(bank, prob=1)
    transform = Compose(spatial_transforms() + [synth])
    dataset = CacheDataset([record], transform, cache_num=1)
    first, second = dataset[0], dataset[0]
    np.testing.assert_allclose(nib.affines.voxel_sizes(first['image'].affine.numpy()), (1, 1, 1))
    assert first['image'].shape[-1] > label.shape[-1] * 2
    assert (first['label'] == 2).any() and (second['label'] == 2).any()
    assert not torch.equal(first['label'], second['label'])
    assert dataset._cache[0]['label'].max() == 1
    # Exercise the real crop/augmentation/collation path.
    patches = Dataset([record], get_train_transform(bank))
    batch = next(iter(DataLoader(patches, batch_size=1, collate_fn=list_data_collate)))
    assert batch['image'].shape == (1, 1, 96, 96, 96)
    assert set(torch.unique(batch['label']).tolist()) <= {0., 1., 2.}


def test_reference_splits_and_explicit_missing_data(tmp_path):
    full = build_splits(tmp_path, 'reference', check_files=False)
    small = build_splits(tmp_path, 'lits_only', check_files=False)
    assert [len(full[k]) for k in ('training', 'validation', 'holdout')] == [116, 22, 98]
    assert len(small['training']) == 11
    sets = [{r['image'] for r in full[k]} for k in ('training', 'validation', 'holdout')]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    assert full['validation'] == small['validation']
    with pytest.raises(FileNotFoundError):
        build_splits(tmp_path)


def test_empty_class_policy_and_physical_nsd():
    target = np.zeros((12, 12, 12), np.uint8)
    pred = target.copy()
    assert np.isnan(native_metrics(pred, target, (1, 1, 1))['tumor_dice'])
    target[3:7, 3:7, 3:7] = 2
    assert native_metrics(pred, target, (1, 1, 1))['tumor_nsd'] == 0
    assert native_metrics(target, target, (1, 1, 1))['tumor_nsd'] == 1
    pred[3:7, 3:7, 4:8] = 2
    assert native_metrics(pred, target, (1, 1, 1))['tumor_nsd'] > native_metrics(pred, target, (1, 1, 5))['tumor_nsd']
    assert validation_dice(np.zeros_like(pred), np.zeros_like(pred), pred.shape) == [0, 0]
    assert denoise_prediction(pred).max() == 0  # no liver: no tumors retained


class OracleModel(UNet3DLightning):
    def infer(self, image):
        # Intensity-coded fixture, independent of the native target and inverse transform.
        centers = torch.tensor([0., 101 / 210, 171 / 210], device=image.device)[None, :, None, None, None]
        return -(image - centers).square()


class NativeTestData(LightningDataModule):
    def __init__(self, record):
        super().__init__()
        self.record = record
        self.test_transform = get_test_transform()

    def test_dataloader(self):
        return DataLoader(Dataset([self.record], self.test_transform), batch_size=1)


def test_native_inversion_and_export_end_to_end(tmp_path):
    record, affine, label = write_case(tmp_path)
    model = OracleModel(prediction_dir=str(tmp_path / 'predictions'))
    trainer = Trainer(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False)
    result = trainer.test(model, datamodule=NativeTestData(record))[0]
    saved = nib.load(tmp_path / 'predictions/label.nii.gz')
    np.testing.assert_allclose(saved.affine, affine)
    assert saved.shape == label.shape
    assert result['test_tumor_dice'] > .95
    assert result['test_tumor_nsd'] > .95
    assert (tmp_path / 'predictions/metrics.csv').exists()


def test_real_unet_backward_and_scheduler():
    model = UNet3DLightning()
    batch = torch.rand(1, 1, 32, 32, 32)
    labels = torch.zeros_like(batch)
    labels[:, :, 8:24, 8:24, 8:24] = 1
    labels[:, :, 12:20, 12:20, 12:20] = 2
    loss = model.loss_fn(model(batch), labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    config = model.configure_optimizers()
    optimizer, scheduler = config['optimizer'], config['lr_scheduler']['scheduler']
    assert optimizer.param_groups[0]['lr'] == 0
    for _ in range(100):
        optimizer.step(); scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(4e-4)
    for _ in range(3900):
        optimizer.step(); scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0, abs=1e-12)


def test_notebook_is_valid_and_has_no_stale_outputs():
    path = Path(__file__).resolve().parents[1] / 'SynTumor.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, str(path), 'exec')
            assert not cell.outputs


def test_lightning_fit_validation_checkpoint_reload(tmp_path):
    from lightning.pytorch.callbacks import ModelCheckpoint
    from syntumor.data import get_val_transform
    record, _, _ = write_case(tmp_path)
    train_batch = {'image': torch.rand(1, 32, 32, 32),
                   'label': torch.zeros(1, 32, 32, 32)}
    train_batch['label'][:, 8:24, 8:24, 8:24] = 1
    train_batch['label'][:, 12:20, 12:20, 12:20] = 2
    callback = ModelCheckpoint(dirpath=tmp_path / 'checkpoints', monitor='val_mean_dice',
                               mode='max', save_last=True)
    model = UNet3DLightning(max_epochs=4, warmup_epochs=2,
                            prediction_dir=str(tmp_path / 'predictions'))
    trainer = Trainer(max_epochs=1, accelerator='cpu', devices=1, logger=False,
                      callbacks=[callback], num_sanity_val_steps=0, enable_progress_bar=False)
    trainer.fit(model, train_dataloaders=DataLoader([train_batch], batch_size=1),
                val_dataloaders=DataLoader(Dataset([record], get_val_transform()), batch_size=1))
    assert trainer.global_step == 1
    assert {'val_liver_dice', 'val_tumor_dice', 'val_mean_dice'} <= trainer.callback_metrics.keys()
    assert Path(callback.best_model_path).is_file()
    restored = UNet3DLightning.load_from_checkpoint(callback.best_model_path)
    for original, reloaded in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(original, reloaded)
    checkpoint = torch.load(callback.last_model_path, weights_only=False)
    assert checkpoint['optimizer_states'] and checkpoint['lr_schedulers']
