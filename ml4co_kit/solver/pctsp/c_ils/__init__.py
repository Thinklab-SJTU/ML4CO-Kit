import os
import uuid
import pathlib
import subprocess
import numpy as np


BASE_PATH = pathlib.Path(__file__).parent
SOLVER_PATH = pathlib.Path(__file__).parent / "pctsp_ils"
TMP_PATH = pathlib.Path(__file__).parent / "tmp"

# Determining whether the solvers have been built
if not os.path.exists(SOLVER_PATH):
    ori_dir = os.getcwd()
    os.chdir(BASE_PATH)
    os.system("make")
    os.chdir(ori_dir)


# make tmp dir
if not os.path.exists(TMP_PATH):
    os.makedirs(TMP_PATH)


def pctsp_ils_solve(
    dist_matrix: np.ndarray, penalties: np.ndarray, prizes: np.ndarray, 
    min_prize_scaled: int, runs_per_instance: int
) -> np.ndarray:
    tmp_name = uuid.uuid4().hex[:9]
    tmp_file_path = TMP_PATH / f"{tmp_name}.txt"
    
    # Write the properly scaled integer data to a tmporary file
    with open(tmp_file_path, 'w') as f:
        # Prizes
        f.write(' '.join(map(str, prizes)) + '\n')
        # penalties
        f.write(' '.join(map(str, penalties)) + '\n')
        # Distance matrix
        for row in dist_matrix:
            f.write(' '.join(map(str, row)) + '\n')
    
    
    # Call the C++ solver with the scaled min_prize
    command = [
        SOLVER_PATH,
        tmp_file_path,
        str(min_prize_scaled), # Use the scaled integer value for min_prize
        str(runs_per_instance)
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, encoding='utf-8'
    )
    output = result.stdout    
    
    # Process the output of the C++ solver
    tour = None
    for line in output.strip().split('\n'):
        if line.startswith("Best Result Route:"):
            full_route = [int(node) for node in line.split(':')[1].strip().split()]
            if len(full_route) > 2 and full_route[0] == 0 and full_route[-1] == 0:
                tour = np.array(full_route)
            else:
                raise RuntimeError("Failed to solve a route from C++ solver output.")
    if tour is None:
        raise RuntimeError("Failed to solve a route from C++ solver output.")

    # Clean up the tmporary input file
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)

    return tour

    