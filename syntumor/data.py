from pathlib import Path
import json
import random
from contextlib import contextmanager

import numpy as np
import torch
import lightning.pytorch as pl
from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, SpatialPadd, EnsureTyped, MapTransform, RandomizableTransform,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
)
from upstream_reference.TumorGenerated.utils import SynthesisTumor, get_predefined_texture

REFERENCE = Path(__file__).resolve().parents[1] / 'upstream_reference'
TEXTURE_SHAPE = (420, 300, 320)
TEXTURE_SIGMAS = [(a, b) for a in (3, 6, 9, 12, 15) for b in (4, 7)]


@contextmanager
def isolated_rng(seed):
    """Adapt upstream global RNG calls without corrupting MONAI/worker RNG streams."""
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class TextureBank:
    """The reference's ten float64 textures, disk-backed to avoid worker copies."""
    def __init__(self, directory='data/reference_textures'):
        self.directory = Path(directory)

    def paths(self):
        return [self.directory / f'texture_{a}_{b}.npy' for a, b in TEXTURE_SIGMAS]

    def prepare(self):
        paths = self.paths()
        if all(p.exists() for p in paths):
            for p in paths:
                t = np.load(p, mmap_mode='r')
                if t.shape != TEXTURE_SHAPE or t.dtype != np.float64:
                    raise ValueError(f'Invalid reference texture: {p}')
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        # Rebuild the full sequence if incomplete, preserving the reference seed/order.
        with isolated_rng(0):
            for (a, b), path in zip(TEXTURE_SIGMAS, paths):
                print(f'Generating reference texture sigma=({a}, {b})', flush=True)
                texture = get_predefined_texture(TEXTURE_SHAPE, a, b)
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as f:
                    np.save(f, texture)
                temporary.replace(path)
                del texture

    def __getitem__(self, index):
        return np.load(self.paths()[index], mmap_mode='r')


class GenerateSyntheticTumor(RandomizableTransform, MapTransform):
    def __init__(self, texture_bank, prob=0.9):
        MapTransform.__init__(self, ('image', 'label'))
        RandomizableTransform.__init__(self, prob)
        self.texture_bank = texture_bank
        self.tumor_types = ('tiny', 'small', 'medium', 'large', 'mix')

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d
        image, label = d['image'], d['label']
        labels = label[0].detach().cpu().numpy()
        if labels.max() > 1:  # Reference does not insert into already diseased livers.
            return d
        if not np.any(labels == 1):
            raise ValueError('Synthetic training requires a nonempty binary liver mask.')
        if not np.all(np.isin(labels, (0, 1))):
            raise ValueError('Expected binary liver labels 0/1.')
        texture = self.texture_bank[int(self.R.randint(10))]
        tumor_type = str(self.R.choice(self.tumor_types))
        # Clone before mutation: cached healthy images must stay healthy.
        volume = image[0].detach().cpu().numpy().copy()
        mask = labels.astype(np.uint8, copy=True)
        bounds = np.argwhere(mask).ptp(axis=0) - 2
        if np.any(bounds <= 0) or np.any(bounds >= np.asarray(texture.shape)):
            raise ValueError('Liver ROI is empty or too large for the reference texture bank.')
        with isolated_rng(self.R.randint(2**32 - 1)):
            volume, mask = SynthesisTumor(volume, mask, tumor_type, texture)
        # Assignment into cloned MetaTensors retains affine and transform history.
        d['image'], d['label'] = image.clone(), label.clone()
        d['image'][0] = torch.as_tensor(volume, dtype=image.dtype, device=image.device)
        d['label'][0] = torch.as_tensor(mask, dtype=label.dtype, device=label.device)
        return d


def spatial_transforms(keys=('image', 'label')):
    return [
        LoadImaged(keys=('image', 'label')),
        EnsureChannelFirstd(keys=('image', 'label'), channel_dim='no_channel'),
        Orientationd(keys=keys, axcodes='RAS', labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
        Spacingd(keys=keys, pixdim=(1., 1., 1.),
                 mode=('bilinear', 'nearest') if len(keys) == 2 else 'bilinear'),
    ]


def intensity_and_pad(keys=('image', 'label')):
    return [
        ScaleIntensityRanged(keys='image', a_min=-21, a_max=189,
                             b_min=0., b_max=1., clip=True),
        SpatialPadd(keys=keys, spatial_size=(96, 96, 96),
                    mode=('minimum', 'constant') if len(keys) == 2 else 'minimum'),
        EnsureTyped(keys=('image', 'label'), dtype=(torch.float32, torch.float32), track_meta=True),
    ]


def get_train_transform(texture_bank):
    return Compose(spatial_transforms() + [GenerateSyntheticTumor(texture_bank)] +
                   intensity_and_pad() + [
        RandCropByPosNegLabeld(keys=('image', 'label'), label_key='label',
                              spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=1,
                              image_key='image', image_threshold=0),
        *[RandFlipd(keys=('image', 'label'), prob=0.2, spatial_axis=i) for i in range(3)],
        RandRotate90d(keys=('image', 'label'), prob=0.2, max_k=3),
        RandScaleIntensityd(keys='image', factors=0.1, prob=0.15),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=0.15),
    ])


def get_val_transform():
    return Compose(spatial_transforms() + intensity_and_pad())


def get_test_transform():
    # Original evaluation keeps the target in its native grid; predictions are inverted.
    return Compose(spatial_transforms(keys=('image',)) + intensity_and_pad(keys=('image',)))


def build_splits(data_root='data', protocol='reference', check_files=True):
    if protocol not in ('reference', 'lits_only'):
        raise ValueError('protocol must be reference or lits_only')
    root = Path(data_root)
    manifest = json.loads((REFERENCE / 'datafolds/healthy.json').read_text())
    train = manifest['training']
    if protocol == 'lits_only':
        train = [r for r in train if r['image'].startswith('04_LiTS/')]
    def resolve(records):
        result = []
        for record in records:
            record = dict(record)
            # LiTS labels in the project already have binary liver labels for healthy cases.
            if protocol == 'lits_only' and record['image'].startswith('04_LiTS/'):
                record['label'] = record['image'].replace('/img/', '/label/')
            result.append({k: str(root / record[k]) for k in ('image', 'label')})
        return result
    splits = {'training': resolve(train), 'validation': resolve(manifest['validation'])}
    # A project extension, explicitly distinct from upstream's validation-set evaluation.
    excluded = {Path(r['image']).name for rows in splits.values() for r in rows}
    splits['holdout'] = [
        {'image': str(root / '04_LiTS/img' / f'liver_{i}.nii.gz'),
         'label': str(root / '04_LiTS/label' / f'liver_{i}.nii.gz')}
        for i in range(131) if f'liver_{i}.nii.gz' not in excluded
    ]
    if check_files:
        require_files(splits['training'] + splits['validation'])
    return splits


def require_files(records):
    missing = [p for row in records for p in row.values() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} required files missing; first entries:\n' +
                                '\n'.join(missing[:8]))


class SyntheticTumorDataModule(pl.LightningDataModule):
    def __init__(self, data_root='data', protocol='reference', batch_size=2,
                 num_workers=0, cache_num=200, texture_dir='data/reference_textures',
                 evaluation_split='validation'):
        super().__init__()
        if evaluation_split not in ('validation', 'holdout'):
            raise ValueError('evaluation_split must be validation or holdout')
        self.data_root, self.protocol = data_root, protocol
        self.batch_size, self.num_workers, self.cache_num = batch_size, num_workers, cache_num
        self.texture_bank = TextureBank(texture_dir)
        self.evaluation_split = evaluation_split
        self.test_transform = get_test_transform()

    def setup(self, stage=None):
        splits = build_splits(self.data_root, self.protocol, check_files=False)
        if stage in (None, 'fit'):
            require_files(splits['training'] + splits['validation'])
            self.texture_bank.prepare()
            self.train_ds = CacheDataset(splits['training'], get_train_transform(self.texture_bank),
                                         cache_num=self.cache_num, num_workers=1, copy_cache=True)
            self.val_ds = Dataset(splits['validation'], get_val_transform())
        if stage in (None, 'test'):
            require_files(splits[self.evaluation_split])
            self.test_ds = Dataset(splits[self.evaluation_split], self.test_transform)

    def loader(self, ds, training=False):
        return DataLoader(ds, batch_size=self.batch_size if training else 1,
                          shuffle=training, num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0,
                          pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)

    def train_dataloader(self):
        return self.loader(self.train_ds, True)

    def val_dataloader(self):
        return self.loader(self.val_ds)

    def test_dataloader(self):
        return self.loader(self.test_ds)
