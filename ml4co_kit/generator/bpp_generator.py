import numpy as np
from ml4co_kit.generator import GeneratorBase

class BPPGenerator(GeneratorBase):
    """
    Generator for Bin Packing Problem instances. Supports uniform or lognormal item size distributions and multiple bin types.
    """
    def __init__(self, 
                 item_count_range=(5, 20), 
                 item_size_dist="uniform", 
                 item_size_params=None, 
                 bin_type_count_range=(1, 3), 
                 bin_capacity_range=(50, 200), 
                 bin_capacity_dist="uniform"):
        """
        :param item_count_range: Tuple (min_items, max_items) for the number of items per instance.
        :param item_size_dist: Distribution for item sizes ("uniform" or "lognormal").
        :param item_size_params: Parameters for item size distribution.
               If uniform, provide {"low": a, "high": b}. 
               If lognormal, provide {"mean": m, "sigma": s} for underlying normal.
               Defaults: uniform in [1, 50] or lognormal with mean=log(10), sigma=1.
        :param bin_type_count_range: Tuple for range of number of bin types per instance.
        :param bin_capacity_range: Tuple (min_cap, max_cap) for bin capacity values.
        :param bin_capacity_dist: Distribution for bin capacities ("uniform" or "lognormal").
        """
        self.item_count_range = item_count_range
        self.item_size_dist = item_size_dist
        self.item_size_params = item_size_params or {}
        self.bin_type_count_range = bin_type_count_range
        self.bin_capacity_range = bin_capacity_range
        self.bin_capacity_dist = bin_capacity_dist

        # Set default distribution parameters if not provided
        if self.item_size_dist == "uniform":
            self.item_size_params.setdefault("low", 1)
            self.item_size_params.setdefault("high", 50)
        elif self.item_size_dist == "lognormal":
            self.item_size_params.setdefault("mean", 2.3)  # log(10) ~ 2.3
            self.item_size_params.setdefault("sigma", 1.0)
        if self.bin_capacity_dist == "uniform":
            self.min_cap = bin_capacity_range[0]
            self.max_cap = bin_capacity_range[1]
        elif self.bin_capacity_dist == "lognormal":
            self.bin_cap_mean = math.log((bin_capacity_range[0] + bin_capacity_range[1]) / 2.0)
            self.bin_cap_sigma = 0.5

    def generate_instance(self):
        """
        Generate a single random BPP instance.
        :return: BPPTask with generated items and bins.
        """
        n_items = np.random.randint(self.item_count_range[0], self.item_count_range[1] + 1)
        n_bin_types = np.random.randint(self.bin_type_count_range[0], self.bin_type_count_range[1] + 1)
        if self.bin_capacity_dist == "uniform":
            bins = list(np.random.randint(self.bin_capacity_range[0], self.bin_capacity_range[1] + 1, size=n_bin_types))
        else:
            caps = np.random.lognormal(mean=self.bin_cap_mean, sigma=self.bin_cap_sigma, size=n_bin_types)
            caps = np.clip(caps, self.bin_capacity_range[0], self.bin_capacity_range[1])
            bins = list(np.rint(caps).astype(int))
        max_bin_cap = max(bins)
        if self.item_size_dist == "uniform":
            low = self.item_size_params["low"]
            high = min(self.item_size_params["high"], max_bin_cap)
            items = list(np.random.randint(low, high + 1, size=n_items))
        else:  
            mean = self.item_size_params["mean"]
            sigma = self.item_size_params["sigma"]
            vols = np.random.lognormal(mean=mean, sigma=sigma, size=n_items)
            if max_bin_cap > 0:
                scale_factor = max_bin_cap / (np.mean(vols) * 2)
            else:
                scale_factor = 1.0
            vols = vols * scale_factor
            vols = np.clip(vols, 1, max_bin_cap)
            items = list(np.rint(vols).astype(int))
        max_item = max(items) if items else 0
        if max_item > max_bin_cap:
            # If an item is too large, increase the largest bin capacity to accommodate
            max_bin_cap = max_item
            bins.append(max_bin_cap)
        return BPPTask(items, bins)

    def generate(self, num_instances, output_file=None):
        """
        Generate multiple BPP instances.
        :param num_instances: Number of instances to generate.
        :param output_file: Optional path to save the instances via BPPWrapper.
        :return: List of BPPTask instances (if output_file not provided).
        """
        instances = [self.generate_instance() for _ in range(num_instances)]
        if output_file:
            wrapper = BPPWrapper()  
            wrapper.dump(instances, output_file)
        return instances
