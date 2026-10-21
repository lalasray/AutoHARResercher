"""Standalone worker for the upstream IMUGPT environment (Python 3.8+).

Uses upstream models, inverse kinematics and sensor.py; no MCP dependency here.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys


def run(config, request, output):
    import numpy as np
    import torch

    root = Path(config['repository']).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import clip
    import models.vqvae as vqvae
    import models.t2m_trans as trans
    from options.option_transformer import get_args_parser
    from utils.motion_process import recover_from_ric
    from Motion.InverseKinematics import animation_from_positions
    from Motion import BVH
    import sensor

    random.seed(request['seed'])
    np.random.seed(request['seed'])
    torch.manual_seed(request['seed'])
    device = torch.device(config.get('device', 'cuda'))
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The official IMUGPT quantizer requires a CUDA device')
    torch.cuda.set_device(device)
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        args = get_args_parser()
    finally:
        sys.argv = saved
    args.dataname, args.down_t, args.depth, args.block_size = 't2m', 2, 3, 51
    encoder, _ = clip.load('ViT-B/32', device=device, jit=False)
    encoder.eval()
    net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim, args.output_emb_width,
                         args.down_t, args.stride_t, args.width, args.depth, args.dilation_growth_rate)
    transformer = trans.Text2Motion_Transformer(num_vq=args.nb_code, embed_dim=1024,
        clip_dim=args.clip_dim, block_size=51, num_layers=9, n_head=16,
        drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate)
    # Only owner-provided official checkpoints are loaded in this isolated environment.
    net.load_state_dict(torch.load(config['vq_checkpoint'], map_location='cpu')['net'], strict=True)
    transformer.load_state_dict(torch.load(config['motion_checkpoint'], map_location='cpu')['trans'], strict=True)
    net.to(device).eval()
    transformer.to(device).eval()
    mean = torch.as_tensor(np.load(config['mean']), device=device)
    std = torch.as_tensor(np.load(config['std']), device=device)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    names = np.array(['Pelvis', 'Left_hip', 'Right_hip', 'Spine1', 'Left_knee', 'Right_knee',
        'Spine2', 'Left_ankle', 'Right_ankle', 'Spine3', 'Left_foot', 'Right_foot', 'Neck',
        'Left_collar', 'Right_collar', 'Head', 'Left_shoulder', 'Right_shoulder',
        'Left_elbow', 'Right_elbow', 'Left_wrist', 'Right_wrist'])
    sensor.version = 'ideal'
    sensor._samplingPeriod = 1.0 / request['sampling_hz']
    work = Path(output).parent
    samples = []
    rng = np.random.default_rng(request['seed'])
    with torch.no_grad():
        feature = encoder.encode_text(clip.tokenize([request['prompt']], truncate=False).to(device)).float()
        for attempt in range(request['count'] * config.get('attempts_per_sample', 10)):
            try:
                # Categorical sampling is an explicit adaptation: upstream greedy decoding
                # otherwise returns the same motion for every repetition of a prompt.
                tokens = transformer.sample(feature, True)
            except UnboundLocalError:  # upstream EOS before the first motion token
                continue
            if tokens.numel() < 3:
                continue
            pose = net.forward_decoder(tokens)
            xyz = recover_from_ric((pose * std + mean).float(), 22).reshape(-1, 22, 3).cpu().numpy()
            xyz[:, :, 1] -= xyz[:, :, 1].min()
            xyz *= config.get('body_scale', 1.7 / 1.55)
            animation, order, _ = animation_from_positions(xyz, parents)
            bvh = str(work / 'motion.bvh')
            BVH.save(bvh, animation, names=names[order], frametime=1.0 / 20)
            values = sensor.extract_vir_imu(bvh, str(work / 'acc.npz'), str(work / 'gyro.npz'), config['sensor_dataset'])
            columns = [np.asarray(values[c['kind']][c['joint']])[:, c['axis']] * c.get('scale', 1)
                       for c in config['channel_map']]
            length = min(map(len, columns))
            if length < request['window_steps']:
                continue
            sequence = np.stack([c[:length] for c in columns], axis=1)
            if not np.isfinite(sequence).all():
                raise ValueError('IMUSim produced non-finite sensor values')
            start = int(rng.integers(length - request['window_steps'] + 1))
            samples.append(sequence[start:start + request['window_steps']])
            if len(samples) == request['count']:
                np.save(output, np.asarray(samples, dtype=np.float32))
                return
    raise RuntimeError('Not enough sufficiently long motions; reduce window length or revise prompt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), json.loads(Path(args.request).read_text()), args.output)


if __name__ == '__main__':
    main()
