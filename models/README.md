# Local model configuration

Set these variables in `settings.py` or `local_config.py`:

- `EMBEDDING_MODEL`: `BAAI/bge-base-en-v1.5` or a local directory.
- `T5_MODEL_PATH`: `t5-base` or a local directory.
- `VQ_WEIGHTS_PATH`: local trained V10 checkpoint; the default location is
  `models/T5/vqvae_t5_v10_epoch10.pt`.
- `DEVICE`: `cuda` or `cpu`.

The standard embedding index is 768-dimensional. The V10 architecture uses
1024 codebook entries, stride 1, commitment cost 0.10 and EMA decay 0.99.
Training and inference share `src.utils.vqvae_t5_v10.VQVAE_T5`.

Model files are stored locally and are excluded from Git and source archives.
