import time
import numpy as np
from tqdm import tqdm
from typing import Union, Any
from pytorch_lightning.utilities import rank_zero_info
from ml4co_kit.solver import TSPSolver
from ml4co_kit.learning.model import BaseModel, ModelConfigurator
from ml4co_kit.learning.search import SearchConfigurator
from model import TSPGNN
from search import tsp_greedy, tsp_2opt


class TSPNARSolver(TSPSolver):
    """
    This is an TSP solver example to easily develop a solver class based on the learning models
    by inheriting from the class ``solver.TSPSolver``.
    """

    def __init__(self):
        super(TSPNARSolver, self).__init__()

    def solve(
        self,
        batch_size: int = 16,
        encoder: Union[BaseModel, str] = "gnn",
        encoder_kwargs: dict = {},
        decoding_type: Union[Any, str] = "greedy",
        decoding_kwargs: dict = {},
        local_search_type: str = "2opt",
        ls_kwargs: dict = {},
        pretrained: bool = True,
        device="cpu",
    ):
        self.decoding_type = decoding_type
        self.ls_type = local_search_type
        self.model_config = ModelConfigurator(model_class={("tsp", "gnn"): TSPGNN})
        self.search_config = SearchConfigurator(
            decoding_class={("tsp", "greedy"): tsp_greedy},
            local_search_class={("tsp", "2opt"): tsp_2opt},
        )

        # encoder & gain heatmap
        if type(encoder) == str:
            encoder_kwargs.update(
                {
                    "mode": "solve",
                    "nodes_num": self.nodes_num,
                }
            )
            self.encoder = self.model_config.get_model(task="tsp", model_name=encoder)(
                **encoder_kwargs
            )
        else:
            self.encoder = encoder
        rank_zero_info(f"Begin encoding, Using {self.encoder}")
        if pretrained:
            rank_zero_info(f"Loading Weights from Pretrained CheckPoint")
            pretrained_path = (
                encoder_kwargs["pretrained_path"]
                if "pretrained_path" in encoder_kwargs.keys()
                else None
            )
            self.encoder.load_weights(pretrained_path)
        self.encoder.to(device)

        # model inference
        solve_begin_time = time.time()
        edge_index = None
        points = self.points
        heatmap = self.encoder.solve(points, edge_index, batch_size, device)
        solve_end_time = time.time()
        solve_time = solve_end_time - solve_begin_time
        rank_zero_info(f"Model Inference, Using {solve_time}")

        # decoding
        if type(decoding_type) == str:
            self.decoding_func = self.search_config.get_decoding_func(
                task="tsp", name=decoding_type
            )
        else:
            self.decoding_func = decoding_type
        rank_zero_info(f"Decoding, Using {self.decoding_func.__name__}")
        decoded_tours = list()
        for idx in tqdm(range(self.points.shape[0]), desc="Decoding"):
            adj_mat = np.expand_dims(heatmap[idx], axis=0)
            tour = self.decoding_func(
                adj_mat=adj_mat, 
                np_points=self.points[idx], 
                edge_index_np=None,
                **decoding_kwargs
            )
            decoded_tours.append(tour[0])
        decoded_tours = np.array(decoded_tours)

        # local_search
        ls_tours = None
        self.local_search_func = self.search_config.get_local_search_func(
            task="tsp", name=local_search_type
        )
        if self.local_search_func is not None:
            rank_zero_info(f"Local Search, Using {self.local_search_func.__name__}")
            ls_tours = list()
            for idx in tqdm(range(self.points.shape[0]), desc="Local Search"):
                adj_mat = heatmap[idx]
                tour = self.local_search_func(
                    np_points=self.points[idx],
                    tours=decoded_tours[idx],
                    adj_mat=adj_mat,
                    device=device,
                    **ls_kwargs,
                )
                ls_tours.append(tour)
            ls_tours = np.array(ls_tours)

        tours = decoded_tours if ls_tours is None else ls_tours
        self.tours = tours
        return tours

    def __repr__(self):
        message = f"encoder={self.encoder}, decoding_type={self.decoding_type}, ls_type={self.ls_type}"
        return f"{self.__class__.__name__}({message})"
