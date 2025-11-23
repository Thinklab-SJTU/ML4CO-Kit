import math
from ml4co_kit.task import Task  

class BPPTask(Task):
    """
    Bin Packing Problem (BPP) Task definition. Contains items and bin types, 
    and methods to validate and evaluate a packing solution.
    """
    def __init__(self, items, bins, label=None):
        """
        Initialize a BPPTask instance.
        :param items: List of item sizes (volumes).
        :param bins: List of bin capacities available (multiple bin types allowed).
        :param label: Optional output label (e.g., best heuristic index) for this instance.
        """
        self.items = list(items)
        self.bins = list(bins)
        self.label = label  # e.g., index of best heuristic, if known
        # Optionally, ensure bins sorted (not strictly necessary, but for clarity)
        self.bins.sort()

    def check_solution(self, solution):
        """
        Check if a given packing solution is feasible (all items packed without exceeding bin capacities).
        :param solution: Solution represented as list of tuples (bin_capacity, [item_indices]).
        :return: True if solution is feasible and uses valid bins, False otherwise.
        """
        n_items = len(self.items)
        used_indices = set()
        for (capacity, item_indices) in solution:
            if capacity not in self.bins:
                return False
            total_vol = 0
            for idx in item_indices:
                if idx < 0 or idx >= n_items or idx in used_indices:
                    return False
                used_indices.add(idx)
                total_vol += self.items[idx]
            if total_vol > capacity:
                return False
        if used_indices != set(range(n_items)):
            return False
        return True

    def evaluate_solution(self, solution):
        """
        Evaluate the packing solution by computing its cost (e.g., number of bins used).
        :param solution: Solution as list of (bin_capacity, [item_indices]).
        :return: The cost of the solution (number of bins used by default).
        """
        if not self.check_solution(solution):
            raise ValueError("Invalid solution provided.")
        return len(solution)
