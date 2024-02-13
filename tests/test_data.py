import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from data4co.data import TSPLIBDataset


def test_tsplib():
    TSPLIBDataset()


if __name__ == "__main__":
    test_tsplib()