from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = [
    'community_partition',
    'roles_and_resampling',
    'community_metrics',
    'partition_stability',
    'block_profile_stability',
    'community_permutations',
    'natural_connectivity',
    'sample_metrics',
    'plant_metrics',
    'observed_fingerprints',
    'high_load_fingerprints',
    'damage_standard_retention',
    'summarize_standard_damage',
    'damage_high_load',
    'summarize_high_load_damage',
    'damage_six_dimensions',
    'magnitude_profile',
    'swi',
    'swi_cross_validation',
    'wsbm_curve',
]

def compute():
    for stage in STAGES:
        print(stage, flush=True)
        subprocess.run([sys.executable, str(ROOT/'analysis'/f'{stage}.py')], cwd=ROOT, check=True)
    from refresh_sources import refresh
    refresh()
    from verify_results import verify
    verify()

def plot(figure=None):
    from figure_plotting import render
    for item in json.loads((ROOT/'figure_index.json').read_text()):
        path = ROOT/item['directory']
        if figure is None or path.name==figure:
            print(path.name, flush=True)
            render(path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--figure')
    args=parser.parse_args()
    if not args.plots_only:
        compute()
    plot(args.figure)
