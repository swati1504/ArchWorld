"""Render the saved reference cloud using a comparison's exact camera and bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_verified_comparison import render

parser = argparse.ArgumentParser()
parser.add_argument('--geometry', type=Path, required=True)
parser.add_argument('--figure', type=Path, required=True)
args = parser.parse_args()
path = args.geometry / 'ground_truth.npz'
cloud = np.load(path, allow_pickle=False)
provenance = json.loads((args.figure / 'render_provenance.json').read_text(encoding='utf-8'))
image, fraction = render(cloud['points'], cloud['colors'], np.array(provenance['basis']),
                         np.array(provenance['shared_bounds']), 1500, 700)
image.save(args.figure / 'reference.png')
(args.figure / 'reference_provenance.json').write_text(json.dumps({
    'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'in_frame_fraction': fraction, 'camera_and_bounds': 'Identical to before_after.png'
}, indent=2), encoding='utf-8')
print('Reference rendered:', fraction, 'of sampled points in frame')
