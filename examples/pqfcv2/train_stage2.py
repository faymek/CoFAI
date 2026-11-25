#!/usr/bin/env python3
"""Train only token transforms using a frozen stage-1 PQFCv2 artifact."""
import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.pqfcv2.stage2_model import Stage2Model, file_sha256
from examples.pqfcv2.lists import read_ade_names, read_nyu_relpaths, ade_feature_name, nyu_feature_name


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='YAML file containing baseline training defaults')
    p.add_argument('--stage1', required=True); p.add_argument('--backbone', required=True)
    p.add_argument('--feature-index', action='append', required=True, help='directory of .npy features or indexed-npy-tar-v1 JSON; repeat for each domain')
    p.add_argument('--ade-list', default=str(ROOT / 'examples/vq_ufc/dinov3/data_lists/ade_1w.txt'))
    p.add_argument('--nyu-list', default=str(ROOT / 'examples/vq_ufc/dinov3/data_lists/nyu_train_1w.txt'))
    p.add_argument('--output-dir', required=True); p.add_argument('--latent-tokens', type=int, required=True)
    p.add_argument('--block-tokens', type=int, default=241); p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=4); p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--lmbda', type=float, default=100); p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--group-chunk', type=int, default=8); p.add_argument('--n-val', type=int, default=50)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--max-files', type=int, default=0)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--resume', action='store_true')
    return p


def load_entries(paths):
    result = []
    for path in paths:
        p = Path(path).resolve()
        if p.is_dir():
            files = sorted(p.glob('*.npy'))
            if not files: raise ValueError(f'no .npy features in {p}')
            result.extend((str(f), f.name, 0, f.stat().st_size) for f in files)
            continue
        index = json.loads(p.read_text())
        if index['format'] != 'indexed-npy-tar-v1': raise ValueError(f'unsupported index {p}')
        archive = p.parent / index['archive']
        if not archive.is_file(): raise FileNotFoundError(archive)
        result.extend((str(archive), name, int(entry['offset']), int(entry['size'])) for name, entry in sorted(index['entries'].items()))
    return result


def load_entry(entry, device):
    path, name, offset, size = entry
    with open(path, 'rb') as f:
        f.seek(offset); data = f.read(size)
    x = np.load(io.BytesIO(data), allow_pickle=False)
    if x.ndim != 2 or x.shape[-1] != 1024 or x.shape[0] <= 5 or not np.isfinite(x).all(): raise ValueError(f'bad feature {name}: {x.shape}')
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)


def parse_args(argv=None):
    parser = args_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config')
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        import yaml
        with open(config_args.config, encoding='utf-8') as handle:
            defaults = yaml.safe_load(handle)
        if not isinstance(defaults, dict): raise ValueError('training config must be a mapping')
        allowed = {'block_tokens', 'epochs', 'batch_size', 'lr', 'lmbda', 'temperature', 'group_chunk', 'n_val', 'seed', 'device'}
        unknown = set(defaults) - allowed
        if unknown: raise ValueError(f'unknown training config fields: {sorted(unknown)}')
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main():
    a = parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    entries = load_entries(a.feature_index)
    by_name = {entry[1]: entry for entry in entries}
    names = [ade_feature_name(x) for x in read_ade_names(Path(a.ade_list))] + [Path(nyu_feature_name(x)).stem + '.npy' for x in read_nyu_relpaths(Path(a.nyu_list))]
    if len(by_name) != len(entries) or set(names) != set(by_name): raise ValueError('feature archives do not exactly match the stage-1 image lists')
    entries = [by_name[n] for n in names]
    if not 0 < a.n_val < len(entries): raise ValueError('n-val must leave at least one training image')
    order = np.random.RandomState(a.seed).permutation(len(entries))
    val = [entries[i] for i in order[:a.n_val]]
    train = [entries[i] for i in order[a.n_val:]]
    if a.max_files: train = train[:a.max_files]
    output = Path(a.output_dir); output.mkdir(parents=True, exist_ok=True)
    split = {'seed': a.seed, 'validation': val, 'train': train}
    split_path = output / 'split.json'
    if split_path.exists() and json.loads(split_path.read_text()) != json.loads(json.dumps(split)): raise ValueError('existing split differs; use a new output directory')
    split_path.write_text(json.dumps(split, indent=2))
    model = Stage2Model(a.stage1, a.block_tokens, a.latent_tokens).to(device)
    with safe_open(a.backbone, framework='pt', device='cpu') as f:
        weight, bias = f.get_tensor('norm.weight').to(device), f.get_tensor('norm.bias').to(device)
    for x in (weight, bias): x.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(model.analysis.parameters()) + list(model.synthesis.parameters()), lr=a.lr)
    stage1_hash = file_sha256(a.stage1); backbone_hash = file_sha256(a.backbone)
    first_epoch, best = 0, float('inf')
    if a.resume:
        ck = torch.load(output / 'last.pt', map_location=device, weights_only=False)
        if ck['stage1_sha256'] != stage1_hash or ck['backbone_sha256'] != backbone_hash or ck['block_tokens'] != a.block_tokens or ck['latent_tokens'] != a.latent_tokens: raise ValueError('resume configuration mismatch')
        model.analysis.load_state_dict(ck['analysis']); model.synthesis.load_state_dict(ck['synthesis']); optimizer.load_state_dict(ck['optimizer'])
        first_epoch, best = ck['epoch'], ck['best']
    def loss(x, training):
        if training: y, bits = model.train_forward(x, a.temperature, a.group_chunk)
        else:
            out = model(x); y = out['h_hat']; bits = -torch.log2(out['likelihoods']['pq'].clamp_min(1e-12)).sum()
        teacher = F.layer_norm(x, (model.channels,), weight, bias, 1e-6)
        student = F.layer_norm(y, (model.channels,), weight, bias, 1e-6)
        distortion = (teacher - student).square().sum()
        return bits + a.lmbda * distortion, bits, distortion
    history_path = output / 'history.jsonl'
    for epoch in range(first_epoch, a.epochs):
        model.train(); order = train.copy(); random.Random(a.seed + epoch).shuffle(order)
        totals = np.zeros(3); optimizer.zero_grad(set_to_none=True)
        for i, entry in enumerate(tqdm(order, desc=f'train {epoch+1}')):
            x = load_entry(entry, device); values = loss(x, True)
            batch_count = min(a.batch_size, len(order) - (i // a.batch_size) * a.batch_size)
            (values[0] / batch_count).backward()
            totals += [v.detach().item() for v in values]
            if (i + 1) % a.batch_size == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(list(model.analysis.parameters()) + list(model.synthesis.parameters()), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        model.eval(); vt = np.zeros(3)
        with torch.no_grad():
            for entry in tqdm(val, desc='validate'):
                values = loss(load_entry(entry, device), False); vt += [v.item() for v in values]
        vt /= len(val); totals /= len(order); improved = vt[0] < best; best = min(best, vt[0])
        ck = {'version': 1, 'epoch': epoch+1, 'stage1_sha256': stage1_hash, 'backbone_sha256': backbone_hash, 'block_tokens': a.block_tokens, 'latent_tokens': a.latent_tokens, 'analysis': model.analysis.state_dict(), 'synthesis': model.synthesis.state_dict(), 'optimizer': optimizer.state_dict(), 'best': best, 'args': vars(a)}
        torch.save(ck, output / 'last.pt')
        if improved: torch.save(ck, output / 'best.pt')
        row = {'epoch': epoch+1, 'train_loss_bits_distortion': totals.tolist(), 'val_loss_bits_distortion': vt.tolist(), 'best': best}
        with history_path.open('a') as f: f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

if __name__ == '__main__': main()
