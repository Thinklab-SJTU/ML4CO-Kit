import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from data4co.data import TSPLIBDataset, TSPLKHDataset, TSPConcordeDataset
from data4co.data import SATLIBDataset


def test_tsp_dataset():
    TSPLIBDataset()
    TSPLKHDataset()
    TSPConcordeDataset()


def test_sat_dataset():
    satlib_dataset = SATLIBDataset()
    satlib_dataset.generate_mis_from_sat()
    

if __name__ == "__main__":
    test_tsp_dataset()
    test_sat_dataset()