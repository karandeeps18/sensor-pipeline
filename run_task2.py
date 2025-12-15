import sys
sys.path.insert(0, ".")

from src.streaming import run_streaming_simulation
from src.config import DATA_DIR


def main():
    """Run streaming simulation"""
    run_streaming_simulation(DATA_DIR, num_days=5)

if __name__ == "__main__":
    main()