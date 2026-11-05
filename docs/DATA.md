# Data acquisition

No CT volumes, masks or model weights are distributed with this repository. The Dropbox packages linked by SyntheticTumors are reported unavailable. The following sources provide alternatives; they are **not drop-in replacements** for the old archive layout.

| Dataset | Source | Required material |
|---|---|---|
| BTCV | [Synapse Abdomen](https://www.synapse.org/Synapse:syn3376386) | CT images for the IDs in the manifest; a 30-case subset is insufficient for its 47 BTCV entries |
| TCIA Pancreas-CT | [TCIA collection](https://www.cancerimagingarchive.net/collection/pancreas-ct/) | CT DICOM series via NBIA Data Retriever |
| CHAOS | [Official Zenodo deposit](https://zenodo.org/records/3431873) | CT portion of `CHAOS_Train_Sets.zip`, including liver masks |
| LiTS | [Original challenge](https://competitions.codalab.org/competitions/17094) / [third-party gated mirror](https://huggingface.co/datasets/Novel-BioMedAI/Liver_Segmentation_Datasets/tree/main/LiTS_Liver-Tumor-Segmentation-Challenge) | Both training batches, with liver and tumor annotations; mirror requires account/terms acceptance |
| BTCV / TCIA liver annotations | [Gibson et al., Zenodo](https://zenodo.org/records/1169361) | `label_btcv_multiorgan.tar.gz`, `label_tciapancreasct_multiorgan.tar.gz`; liver is class 6 |

TCIA's standard pancreas masks are not liver annotations. The multiorgan deposit notes potentially incomplete annotations outside its `cropping.csv` regions. Check coverage and image/mask geometry before use. Source pages were inspected; successful authenticated downloads of every archive have not been verified.

## Preparation requirements

1. Match original patient IDs to `upstream_reference/datafolds/healthy.json`; do not assign IDs by file order.
2. Convert DICOM images and slice masks to NIfTI while preserving HU values, slice ordering and physical geometry.
3. Verify image/mask affine, dimensions, labels and overlay alignment.
4. Extract binary liver masks for synthetic training. Preserve LiTS labels 0/1/2 for evaluation.
5. Arrange files according to the manifest below. The repository does not yet include converters for all alternative source formats.

```text
data/
├── 01_Multi-Atlas_Labeling/img/
├── 02_TCIA_Pancreas-CT/img/
├── 03_CHAOS/ct/
├── 04_LiTS/
│   ├── img/
│   └── label/
└── label/   # binary liver masks, named exactly as in healthy.json
```

The manifest is the authoritative source of complete filenames. In `lits_only` mode, healthy masks are read from `04_LiTS/label/` instead of the separate binary-mask directory.

Follow each provider's access conditions and citation requirements. Do not commit datasets to Git.
