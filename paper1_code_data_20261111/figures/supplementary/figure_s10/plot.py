from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from figure_plotting import render

if __name__ == '__main__':
    render(Path(__file__).resolve().parent)
