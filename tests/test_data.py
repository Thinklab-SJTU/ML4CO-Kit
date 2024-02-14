import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from data4co.data import TSPLIBDataset, TSPLKHDataset, TSPConcordeDataset


def test_tsp_dataset():
    TSPLIBDataset()
    TSPLKHDataset()
    TSPConcordeDataset()


if __name__ == "__main__":
    test_tsp_dataset()