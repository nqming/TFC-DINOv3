# Modification report: TF-C + DINOv3

## Scope and provenance

Git cloned existing `nqming/TFC-DINOv3`, added upstream `https://github.com/mims-harvard/TFC-pretraining.git`, and created `feature/dinov3-custom-loss`. Initial origin/main and fetched upstream/main both resolve to `96675826e9ef234a9b01cc63d484c66cb0441bc0`. Original commits, authors and LICENSE remain intact. DINOv3 is a separate official checkout at `6876159a11b4df116f30f667f8c9888617df0751`. No unrelated baseline was edited.

## 1. Original architecture

Official implementation: `code/TFC`, identified in its readme as the January 2023 Transformer update rather than the older CNN. The root README, directory tree, all five dataset Config files, entry point, model, loss, trainer, loader, augmentation and utility imports were inspected. Baseline-specific conda manifests remain unchanged.

The loader selects the first channel, truncates to length L, and returns time, labels, jittered time, FFT magnitude and perturbed frequency. Augmentations are cached at dataset creation. Fine-tuning substitutes original views for augmented views. Frequency perturbation is independent of time jitter.

TFC owns separate time/frequency TransformerEncoders, each with two layers, two attention heads, embedding dimension L and feed-forward dimension 2L. Projectors: Linear(L,256), BatchNorm(256), ReLU, Linear(256,128). Output `(h_time,z_time,h_freq,z_freq)` has shapes `[B,L],[B,128],[B,L],[B,128]`. The classifier concatenates projections, maps 256 to 64, applies sigmoid, and maps to class logits.

The original Transformer is sequence-first but receives `[B,1,L]`, so attention couples samples within a batch. Original mode preserves this for checkpoint/output parity. Correcting that layout should be a separately named experimental baseline. DINOv3 processes each image independently.

loss.py contains NTXentLoss, NTXentLoss_poly and hierarchical_contrastive_loss; the trainer uses only NTXentLoss_poly. Pretraining optimizes `0.2*(loss_t+loss_f)+l_TF`; fine-tuning adds supervised CE with coefficient 0.1 on within-domain losses. loss_c differences are computed but unused. The polynomial correction is unchanged, including its mean across all one-hot/probability entries.

Adam uses dataset LR/betas and weight decay 3e-4. ReduceLROnPlateau is stepped only in fine-tuning. Pretraining originally clears gradients once per epoch, constructs CUDA masks even for CPU inputs and uses malformed logger arguments. Fixes change optimization trajectories even with custom loss disabled.

Original feasibility: upstream model initialization, exact eval parity, original loss/backward and legacy checkpoint loading pass on synthetic tensors. The unmodified CPU data path is blocked by CUDA-only augmentation. Full original experiments were not run without real datasets/checkpoints.

## 2. Improved architecture and decisions

DINOv3TFC replaces both encoders only when selected, inherits TFC.forward and creates original-shaped projectors without allocating discarded Transformers. Each branch owns an official ViT and CLS-to-L feature adapter. ViT-S/16 is tested; B/16 and L/16 are selectable but untested.

Separate differentiable GASF adapters preserve paired samples and domain separation. A combined time/frequency image was avoided because it would eliminate the two-domain objective. README documents exact scaling and equations. GAF follows Wang and Oates (2015), with explicitly different tanh/RMS normalization here.

Official hubconf.py eagerly imports task heads requiring torchvision/torchmetrics. The final loader calls the unmodified factory in `dinov3.hub.backbones`, the same factory used by Hub. No copied/replacement model is used. An already-imported different checkout is rejected. Pretrained mode requires an authorized local checkpoint with no silent random fallback/download.

Freeze controls requires_grad and keeps backbones in eval mode through train() calls. Feature projections and projectors remain trainable. Autograd remains enabled through inputs. Adam includes all parameters, permitting explicit unfreeze without new optimizer groups. Scheduled unfreeze is not implemented.

## 3. Auxiliary loss

RelationalConsistencyLoss matches off-diagonal cosine Gram entries across domains, averaged over original/augmented pairs. This matches inter-sample geometry rather than only paired positives. It is invariant to orthogonal coordinate changes, needs no teacher/labels/stop-gradient and differentiates through both branches. Float32 computation, epsilon-normalization, dimension checks and finite-value checks support stable ordinary use.

`L_total=L_TFC+lambda*L_custom`. Disabled/zero-weight paths return the original loss tensor directly. Collapse also minimizes this auxiliary loss alone, so the retained contrastive objective is essential. README contains full equations and gradient routes.

## 4. File-by-file before/after map

| File | Before | After and reason |
|---|---|---|
| `.gitignore` | Cache/data ignores. | Adds virtualenv, pytest cache, weight formats and logs. Keep local dependencies and large weights out of Git. |
| `code/TFC/augmentations.py` | CUDA allocations in remove_frequency/add_frequency. | Uses rand_like on input device. Enable CPU/CUDA-compatible frequency augmentation. |
| `code/TFC/main.py` | Direct TFC construction, implicit device, relative data paths, weight-only load. | Config/factory, explicit device, data root, loss logs, encoder run paths and resume; subset becomes opt-in. Expose extensions through the existing CLI. |
| `code/TFC/trainer.py` | Original objective; gradient reset once per epoch; final weight-only checkpoint; global fine-tuning paths. | Optional custom loss, finite checks/logging, per-batch zero_grad, actual batch size, epoch checkpoints, run-local fine-tuning paths, modern NumPy and class-count handling. Preserve objective while fixing training/device/checkpoint issues. |
| `code/TFC/dinov3_encoder.py` | No vision encoder. | GramianAdapter, DINOv3Encoder, DINOv3TFC and build_model with official factories and independent domain backbones. Bridge one-dimensional views to DINOv3 while retaining original forward/projection contract. |
| `code/TFC/custom_loss.py` | No auxiliary relational objective. | RelationalConsistencyLoss and add_custom_loss with validation/zero-weight bypass. Align inter-sample geometry without replacing paired contrastive learning. |
| `code/TFC/checkpoint.py` | Inline weight-only checkpoints. | save_checkpoint/load_checkpoint with metadata, optimizer, epoch and strict legacy loading. Resume pretraining and reject incompatible configurations. |
| `code/config_files/extension.py` | Dataset Python configs only. | DEFAULTS/configure and validated JSON overrides. Share extension controls without modifying inherited dataset configs. |
| `configs/original.json` | Absent. | Original encoder with custom loss disabled. Explicit original-mode recipe. |
| `configs/dinov3.json` | Absent. | ViT-S/16, 224-pixel GASF, random unfrozen weights, batch 8, auxiliary weight 0.1. Explicit no-download starting recipe. |
| `requirements.txt` | Legacy baseline conda environments only. | Modern TF-C/pytest dependency ranges. Install extension without changing unrelated baseline environments. |
| `tests/conftest.py` | Absent. | Root import paths and two CPU threads. Run repository-root tests with bounded CPU overhead. |
| `tests/test_extensions.py` | No extension regression suite. | Original parity, loss math/gradients, actual DINOv3, freeze/unfreeze, checkpoint, CLI/resume and explicit skips. Validate mechanics without claiming benchmark performance. |
| `docs/UPSTREAM_README.md` | Original root README. | Exact content archive; relative links keep their historical root-relative context. Preserve attribution and original usage history. |
| `README.md` | Paper-focused upstream README. | Extension architecture, equations, usage, tests and source map. Make modifications auditable. |
| `docs/MODIFICATION_REPORT.md` | Absent. | Architecture analysis, before/after map, exact diff hunks, tests, limitations and future work. Record engineering decisions and evidence. |

README lists actual class/function ranges. New public symbols: GramianAdapter, DINOv3Encoder, DINOv3TFC, build_model, RelationalConsistencyLoss, add_custom_loss, configure, save_checkpoint and load_checkpoint. model.py/loss.py/dataloader.py and baseline directories remain unchanged.

## 5. Exact Git comparison

```powershell
git diff 96675826e9ef234a9b01cc63d484c66cb0441bc0 CODE_COMMIT_SHA --stat
git diff 96675826e9ef234a9b01cc63d484c66cb0441bc0 CODE_COMMIT_SHA -- code/TFC code/config_files/extension.py configs tests requirements.txt
git diff 96675826e9ef234a9b01cc63d484c66cb0441bc0 CODE_COMMIT_SHA -- LICENSE code/baselines code/TFC/model.py code/TFC/loss.py code/TFC/dataloader.py
git merge-base --is-ancestor 96675826e9ef234a9b01cc63d484c66cb0441bc0 HEAD
```

Third diff is empty; ancestry check succeeds. A documentation-only commit resolves permalinks after the implementation snapshot. Exact changed-line hunks from `git diff --unified=0` before that commit:

- `.gitignore`: `@@ -8,0 +9,6 @@ datasets`
- `code/TFC/augmentations.py`: `@@ -61 +61 @@ def remove_frequency(x, pertub_ratio=0.0):`
- `code/TFC/augmentations.py`: `@@ -67 +67 @@ def add_frequency(x, pertub_ratio=0.0):`
- `code/TFC/augmentations.py`: `@@ -70 +70 @@ def add_frequency(x, pertub_ratio=0.0):`
- `code/TFC/main.py`: `@@ -3,0 +4,7 @@ import os`
- `code/TFC/main.py`: `@@ -25 +32 @@ parser.add_argument('--seed', default=42, type=int, help='seed value')`
- `code/TFC/main.py`: `@@ -39,7 +46,10 @@ parser.add_argument('--home_path', default=home_dir, type=str,`
- `code/TFC/main.py`: `@@ -59 +69,2 @@ exec(f'from config_files.{pretrain_dataset}_Configs import Config as Configs')`
- `code/TFC/main.py`: `@@ -69 +80 @@ np.random.seed(SEED)`
- `code/TFC/main.py`: `@@ -81,0 +93,4 @@ logger = _logger(log_file_name)`
- `code/TFC/main.py`: `@@ -90,3 +105,3 @@ logger.debug("=" * 45)`
- `code/TFC/main.py`: `@@ -98 +113 @@ logger.debug("Data loaded ...")`
- `code/TFC/main.py`: `@@ -106 +121 @@ if training_mode == "fine_tune_test":`
- `code/TFC/main.py`: `@@ -108,3 +123 @@ if training_mode == "fine_tune_test":`
- `code/TFC/main.py`: `@@ -114,0 +128,3 @@ classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.lr,`
- `code/TFC/trainer.py`: `@@ -1,0 +2 @@ import os`
- `code/TFC/trainer.py`: `@@ -5,0 +7,2 @@ from loss import *`
- `code/TFC/trainer.py`: `@@ -26 +29 @@ def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl,`
- `code/TFC/trainer.py`: `@@ -30 +33,4 @@ def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl,`
- `code/TFC/trainer.py`: `@@ -33,2 +39 @@ def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl,`
- `code/TFC/trainer.py`: `@@ -44,0 +50,4 @@ def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl,`
- `code/TFC/trainer.py`: `@@ -55,2 +63,0 @@ def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl,`
- `code/TFC/trainer.py`: `@@ -59,3 +66,2 @@ def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl,`
- `code/TFC/trainer.py`: `@@ -67,2 +73,2 @@ def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl,`
- `code/TFC/trainer.py`: `@@ -114,0 +121 @@ def model_pretrain(model, model_optimizer, criterion, train_loader, config, devi`
- `code/TFC/trainer.py`: `@@ -125 +132 @@ def model_pretrain(model, model_optimizer, criterion, train_loader, config, devi`
- `code/TFC/trainer.py`: `@@ -136 +143,7 @@ def model_pretrain(model, model_optimizer, criterion, train_loader, config, devi`
- `code/TFC/trainer.py`: `@@ -141,0 +155,2 @@ def model_pretrain(model, model_optimizer, criterion, train_loader, config, devi`
- `code/TFC/trainer.py`: `@@ -178 +193 @@ def model_finetune(model, model_optimizer, val_dl, config, device, training_mode`
- `code/TFC/trainer.py`: `@@ -196 +211,7 @@ def model_finetune(model, model_optimizer, val_dl, config, device, training_mode`
- `code/TFC/trainer.py`: `@@ -199 +220 @@ def model_finetune(model, model_optimizer, val_dl, config, device, training_mode`
- `code/TFC/trainer.py`: `@@ -205 +226 @@ def model_finetune(model, model_optimizer, val_dl, config, device, training_mode`
- `code/TFC/trainer.py`: `@@ -269 +290 @@ def model_test(model,  test_dl, config,  device, training_mode, classifier=None,`
- `code/TFC/trainer.py`: `@@ -276 +297 @@ def model_test(model,  test_dl, config,  device, training_mode, classifier=None,`

## 6. Validation evidence

Recorded PowerShell command from the fork root:

```powershell
$env:DINOV3_REPO=(Resolve-Path ../dinov3-official).Path
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=../tfc-test-cli
```

**19 passed, 2 skipped in 36.30s**, 2026-10-29. Python 3.12.14, torch 2.15.0+cpu, NumPy 2.5.4, scikit-learn 1.9.1, SciPy 1.18.1, pandas 3.1.0, pytest 9.2.0. torchvision 0.29.2/Pillow 13.0.0 were installed for Hub diagnosis but are not required by the final loader. Requirements are version ranges, not a complete lockfile.

- Exact original-model eval comparison at length 178 against source loaded via git show and identical state dict; original loss/backward and legacy checkpoint pass.
- Adapter shape, gradients, constant input and invalid-input rejection pass.
- Auxiliary loss matches an independent mathematical reference, propagates gradients, handles zero/disabled weights and rejects NaN/Inf/incompatible dimensions.
- Actual official ViT-S/16 forwards at 32/224 pixels. Unfrozen backward/trainer steps run at 32; both patch embeddings receive finite nonzero gradients. Frozen backbones stay in eval mode without parameter gradients.
- Original/DINO state and optimizer round trips pass; encoder metadata mismatch is rejected.
- Original pretraining/fine-tuning, epoch checkpoint save, and both encoders' CLI pretraining/resume pass. CLI tests use temporary synthetic tensor datasets and advance epoch 1 to 2.
- Pretrained test skips without DINOV3_WEIGHTS; CUDA test skips on this CPU-only runtime.
- compileall, CLI --help and git diff --check (excluding the exact upstream README archive) pass; the archive intentionally retains upstream whitespace. No real-data benchmark/full-resolution backward was performed.

Initial failures exposed eager imports in hubconf.py; final direct factory loading resolves them. Tiny synthetic classification batches can emit undefined-metric warnings; printed scores are not empirical results. Original Transformer layout and NumPy jitter warnings remain documented.

## 7. Technical limitations

1. GASF resampling/normalization lose detail and absolute offset/scale, with sign ambiguities; natural-image transfer is unverified.
2. Original mode preserves batch-coupled attention. Output parity does not establish architectural correctness.
3. Loader discards extra channels, caches augmentations and does not pad short inputs. Inherited configs need review: HAR length 206 can exceed source windows; Epilepsy Config lacks TSlength_aligned for use as a source recipe.
4. Inherited evaluation selects on fine-tuning data, averages some metrics per batch, reports best test scores across epochs, and uses raw logits for AUC/PRC. Redesign around held-out validation before publishing comparisons. This fork does not claim to repair that protocol.
5. Two ViTs increase memory/compute. B/16, L/16, CUDA, released weights, full-resolution backward and dataset-scale fine-tuning are untested.
6. Resume restores model/optimizer/epoch, not RNG/cached augmentations; fine-tuning optimizer/scheduler resume is unsupported. Full model checkpoint files can be large and remain ignored.
7. Training batches need at least two samples for BatchNorm/contrastive learning. Large auxiliary weights may conflict with useful features; tune on validation data with ablations.
8. No external/private application data or real-data superiority claims were introduced.

## 8. Future experiments

Run original-vs-DINOv3 and zero-vs-positive-lambda ablations using identical splits/seeds and a separately versioned corrected original attention baseline. Compare resolutions, amplitude-preserving features, frozen/unfrozen pretrained weights and shared/independent backbones. Add RNG checkpointing and proper validation selection before multi-dataset, multi-seed benchmarks. Report accuracy/F1, transfer, compute and uncertainty only after experiments.

## 9. Attribution

TF-C: Xiang Zhang, Ziyuan Zhao, Theodoros Tsiligkaridis and Marinka Zitnik, NeurIPS 2022. Original MIT LICENSE/copyright remains intact. DINOv3 is Meta's official implementation under its separate license. README contains citations/stable source links; archived README preserves upstream material.
