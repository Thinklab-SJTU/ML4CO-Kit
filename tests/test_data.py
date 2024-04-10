import os
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from data4co.data import TSPLIBOriDataset, TSPUniformDataset
from data4co.data import SATLIBOriDataset


def test_tsp_dataset():
    TSPLIBOriDataset()
    TSPUniformDataset()


def test_sat_dataset():
    SATLIBOriDataset()


if __name__ == "__main__":
    test_tsp_dataset()
    test_sat_dataset()
