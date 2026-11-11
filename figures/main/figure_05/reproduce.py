from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from pipeline import compute,plot

if __name__=='__main__':
    compute()
    plot(Path(__file__).resolve().parent.name)
