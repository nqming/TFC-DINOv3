import logging
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from config_files.SleepEEG_Configs import Config
from config_files.extension import configure
from custom_loss import RelationalConsistencyLoss, add_custom_loss
from dinov3_encoder import GramianAdapter, build_model
from checkpoint import save_checkpoint, load_checkpoint
from dataloader import Load_Dataset
from loss import NTXentLoss_poly
from trainer import Trainer, model_pretrain, model_finetune
from model import target_classifier

BASE = "96675826e9ef234a9b01cc63d484c66cb0441bc0"


def config():
    c = configure(Config())
    c.TSlength_aligned = 32
    c.batch_size = 2
    c.target_batch_size = 2
    c.custom_loss_enabled = True
    c.custom_loss_weight = .1
    return c


def loader(c, mode="pre_train"):
    data = {"samples": torch.randn(6, 1, c.TSlength_aligned), "labels": torch.tensor([0, 1, 0, 1, 0, 1])}
    return DataLoader(Load_Dataset(data, c, mode), batch_size=2)


def test_original_matches_upstream_and_checkpoint(tmp_path):
    c = config()
    c.TSlength_aligned = 178
    source = subprocess.check_output(["git", "show", f"{BASE}:code/TFC/model.py"], text=True)
    namespace = {}
    exec(compile(source, "upstream_model.py", "exec"), namespace)
    upstream = namespace["TFC"](c).eval()
    model = build_model(c).eval()
    model.load_state_dict(upstream.state_dict(), strict=True)
    x = torch.randn(2, 1, 178)
    for original, modified in zip(upstream(x, x), model(x, x)):
        torch.testing.assert_close(original, modified, rtol=0, atol=0)
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": upstream.state_dict()}, path)
    assert load_checkpoint(path, model, "cpu", c) == 0
    h_t, z_t, h_f, z_f = model(x, x + .1)
    loss = NTXentLoss_poly("cpu", 2, .2, True)(z_t, z_f)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.transformer_encoder_t.layers[0].linear1.weight.grad is not None


@pytest.mark.parametrize("constant", [False, True])
def test_adapter_shape_and_gradient(constant):
    x = (torch.zeros(2, 1, 178) if constant else torch.randn(2, 1, 178)).requires_grad_()
    image = GramianAdapter(32)(x)
    assert image.shape == (2, 3, 32, 32)
    image.square().mean().backward()
    assert torch.isfinite(image).all() and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("bad", [torch.randn(2, 3, 32), torch.full((2, 1, 32), float('nan'))])
def test_adapter_rejects_invalid(bad):
    with pytest.raises(ValueError):
        GramianAdapter(32)(bad)


def test_relational_math_backward():
    t = torch.randn(4, 128, requires_grad=True)
    f = torch.randn(4, 128, requires_grad=True)
    criterion = RelationalConsistencyLoss()
    value = criterion(t, f)
    tn, fn = torch.nn.functional.normalize(t, dim=1), torch.nn.functional.normalize(f, dim=1)
    expected = sum(((tn[i] @ tn[j]) - (fn[i] @ fn[j])) ** 2
                   for i in range(4) for j in range(4) if i != j) / 12
    torch.testing.assert_close(value, expected)
    value.backward()
    for z in (t, f):
        assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert criterion(t.detach(), t.detach()) == 0
    assert torch.isfinite(criterion(torch.zeros(4, 128), torch.zeros(4, 128)))


@pytest.mark.parametrize("shape", [(1, 8), (2, 3, 8)])
def test_loss_dimensions(shape):
    with pytest.raises(ValueError):
        RelationalConsistencyLoss()(torch.ones(shape), torch.ones(shape))


@pytest.mark.parametrize("bad", [float('nan'), float('inf')])
def test_loss_rejects_nonfinite(bad):
    with pytest.raises(ValueError):
        RelationalConsistencyLoss()(torch.full((2, 8), bad), torch.ones(2, 8))


@pytest.mark.parametrize("enabled,weight", [(True, 0), (False, .5), (True, .5)])
def test_weight_and_zero_equivalence(enabled, weight):
    c = config()
    c.custom_loss_enabled, c.custom_loss_weight = enabled, weight
    original = torch.tensor(2., requires_grad=True)
    z = [torch.randn(3, 8, requires_grad=True) for _ in range(4)]
    total, auxiliary = add_custom_loss(original, *z, c)
    torch.testing.assert_close(total, original + weight * auxiliary)
    total.backward()
    if not enabled or weight == 0:
        assert total is original and auxiliary == 0
        assert all(x.grad is None for x in z)
    else:
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in z)


def test_original_training_checkpoint_and_resume(tmp_path, caplog):
    c = config()
    model = build_model(c)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.projector_t[0].weight.detach().clone()
    with caplog.at_level(logging.INFO, logger="tfc.loss"):
        value = model_pretrain(model, optimizer, None, loader(c), c, "cpu", "pre_train")
    assert torch.isfinite(value) and not torch.equal(before, model.projector_t[0].weight)
    assert "original=" in caplog.text and "custom=" in caplog.text
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, optimizer, 3, c)
    restored = build_model(c)
    opt = torch.optim.Adam(restored.parameters(), lr=1e-4)
    assert load_checkpoint(path, restored, "cpu", c, opt) == 3
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
    assert len(opt.state) == len(optimizer.state)
    c.encoder = "dinov3"
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(path, restored, "cpu", c)


def test_finetune_step():
    c = config()
    model, classifier = build_model(c), target_classifier(c)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    result = model_finetune(model, optimizer, loader(c, "fine_tune_test"), c, "cpu",
                            "fine_tune_test", classifier, classifier_optimizer)
    assert torch.isfinite(result[0])


def test_trainer_saves_epoch_checkpoint(tmp_path):
    c = config()
    c.num_epoch = 1
    model, classifier = build_model(c), target_classifier(c)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt_c = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    Trainer(model, optimizer, classifier, opt_c, loader(c), None, None, "cpu",
            logging.getLogger("test"), c, str(tmp_path), "pre_train")
    path = tmp_path / "saved_models" / "ckp_last.pt"
    assert load_checkpoint(path, model, "cpu", c, optimizer) == 1


@pytest.fixture
def dino_config():
    repo = os.environ.get("DINOV3_REPO")
    if not repo:
        pytest.skip("Set DINOV3_REPO to official source checkout for real backbone tests")
    c = config()
    c.encoder, c.dinov3_repo, c.dinov3_image_size = "dinov3", repo, 32
    c.dinov3_pretrained, c.dinov3_freeze = False, True
    return c


def test_real_dinov3_forward_freeze_unfreeze_training(dino_config, tmp_path):
    c = dino_config
    model = build_model(c).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in optimizer_ids for p in model.parameters() if p.requires_grad)
    x = torch.randn(2, 1, 32)
    outputs = model(x, x.abs())
    assert [tuple(o.shape) for o in outputs] == [(2,32), (2,128), (2,32), (2,128)]
    for encoder in (model.transformer_encoder_t, model.transformer_encoder_f):
        assert not encoder.backbone.training
        assert all(not p.requires_grad for p in encoder.backbone.parameters())
    sum(o.square().mean() for o in outputs).backward()
    assert model.transformer_encoder_t.feature_projection.weight.grad is not None
    assert all(p.grad is None for p in model.transformer_encoder_t.backbone.parameters())
    for encoder in (model.transformer_encoder_t, model.transformer_encoder_f):
        encoder.set_frozen(False)
        assert encoder.backbone.training
    value = model_pretrain(model, optimizer, None, loader(c), c, "cpu", "pre_train")
    assert torch.isfinite(value)
    for encoder in (model.transformer_encoder_t, model.transformer_encoder_f):
        grad = encoder.backbone.patch_embed.proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    path = tmp_path / "dino.pt"
    save_checkpoint(path, model, optimizer, 1, c)
    assert load_checkpoint(path, model, "cpu", c, optimizer) == 1
    model.eval()
    for encoder in (model.transformer_encoder_t, model.transformer_encoder_f):
        encoder.adapter = GramianAdapter(224)
    with torch.no_grad():
        assert all(torch.isfinite(o).all() for o in model(x, x.abs()))


def test_pretrained_checkpoint(dino_config):
    weights = os.environ.get("DINOV3_WEIGHTS")
    if not weights:
        pytest.skip("Authorized pretrained weights not provided")
    c = dino_config
    c.dinov3_pretrained, c.dinov3_weights = True, weights
    model = build_model(c).eval()
    with torch.no_grad():
        assert all(torch.isfinite(x).all() for x in model(torch.randn(2,1,32), torch.randn(2,1,32)))


def test_cuda_adapter_loss():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(2,1,32, device="cuda", requires_grad=True)
    image = GramianAdapter(32).cuda()(x)
    RelationalConsistencyLoss()(image.flatten(1), image.flip(-1).flatten(1)).backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("encoder", ["original", "dinov3"])
def test_cli_pretraining_and_resume(tmp_path, encoder):
    repo = os.environ.get("DINOV3_REPO")
    if encoder == "dinov3" and not repo:
        pytest.skip("Set DINOV3_REPO for official backbone CLI test")
    root = Path(__file__).resolve().parents[1]
    for dataset in ("SleepEEG", "Epilepsy"):
        folder = tmp_path / "data" / dataset
        folder.mkdir(parents=True)
        for split in ("train", "test"):
            torch.save({"samples": torch.randn(6, 1, 178), "labels": torch.tensor([0,1,0,1,0,1])},
                       folder / f"{split}.pt")
    cfg = dict(encoder=encoder, num_epoch=1, batch_size=2, target_batch_size=2,
               custom_loss_enabled=True, custom_loss_weight=.1, dinov3_image_size=32,
               dinov3_freeze=True)
    if repo:
        cfg["dinov3_repo"] = repo
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    command = [sys.executable, str(root / "code/TFC/main.py"), "--config", str(config_path),
               "--training_mode", "pre_train", "--device", "cpu", "--data_root", str(tmp_path / "data"),
               "--logs_save_dir", str(tmp_path / "logs")]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "custom=" in result.stdout
    paths = list((tmp_path / "logs").rglob("ckp_last.pt"))
    assert len(paths) == 1
    cfg["num_epoch"] = 2
    config_path.write_text(json.dumps(cfg))
    result = subprocess.run(command + ["--resume", str(paths[0])], cwd=root, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert torch.load(paths[0], weights_only=True)["epoch"] == 2
