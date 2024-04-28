import torch
import numpy as np
import torch.nn.functional as F
from ml4co_kit.learning.model import BaseModel
from ml4co_kit.evaluate import TSPEvaluator
from ml4co_kit.learning.search import SearchConfigurator
from env import TSPEnv
from search import tsp_greedy, tsp_2opt
from typing import Any, Union


class TSPGNN(BaseModel):
    def __init__(
        self,
        nodes_num: int,
        # network
        network_type: str = "gnn",
        input_dim: int = 2,
        hidden_dim: int = 256,
        output_channels: int = 2,
        num_layers: int = 12,
        # env
        mode: str = "train",
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        train_batch_size: int = 64,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        valid_samples: int = 1280,
        num_workers: int = 4,
        # training params
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        # test_step
        decoding_type: str = "greedy",
        local_search_type: str = "2opt",
        **kwargs,
    ):
        # network and model
        self.net = self.get_net(network_type)
        model = self.net(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            num_layers=num_layers,
        )

        # env
        env = TSPEnv(
            nodes_num=nodes_num,
            mode=mode,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            valid_samples=valid_samples,
            num_workers=num_workers,
        )

        self.search_config = SearchConfigurator(
            decoding_class={("tsp", "greedy"): tsp_greedy},
            local_search_class={("tsp", "2opt"): tsp_2opt},
        )

        # super
        super(TSPGNN, self).__init__(
            env=env,
            model=model,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # nodes num
        self.nodes_num = nodes_num

        # test
        self.decoding_type = decoding_type
        self.test_decoding_kwargs = kwargs
        self.local_search_type = local_search_type
        self.test_ls_kwargs = kwargs

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        """Shared step between train/val/test. To be implemented."""
        edge_index = None  # for sparse graph input
        np_edge_index = None  # for sparse graph input
        _, points, adj_matrix, ref_tour = batch
        points: torch.Tensor
        ref_tour: torch.Tensor
        if phase != "train":
            np_points = points.cpu().numpy()[0]
            np_ref_tour = ref_tour.cpu().numpy()[0]

        # forward
        x_pred = self.forward(points, edge_index)
        loss = self.loss_fn(x_pred, adj_matrix)

        # return loss if current is a training step
        if phase == "train":
            metrics = {"train/loss": loss}
            for k, v in metrics.items():
                self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            return loss

        # gain heatmap
        heatmap = F.softmax(x_pred, dim=1)
        adj_mat = heatmap[:, 1]

        # decoding / solve
        if phase == "val":
            adj_mat = adj_mat.cpu().numpy()
            solved_tours = tsp_greedy(
                adj_mat=adj_mat,
                np_points=np_points,
                edge_index_np=np_edge_index,
                device=ref_tour.device,
            )
        else:
            # decode
            decoding_func = self.search_config.get_decoding_func(
                task="tsp", name=self.decoding_type
            )
            solved_tours = decoding_func(
                adj_mat=adj_mat,
                np_points=np_points,
                edge_index_np=np_edge_index,
                device=ref_tour.device,
                **self.test_decoding_kwargs,
            )

            # local_search
            local_search_func = self.search_config.get_local_search_func(
                task="tsp", name=self.local_search_type
            )
            solved_tours = local_search_func(
                np_points=np_points,
                tours=solved_tours,
                adj_mat=adj_mat,
                device=ref_tour.device,
                **self.test_ls_kwargs,
            )

        # Check the tours
        for idx in range(len(solved_tours)):
            assert sorted(solved_tours[idx][:-1]) == [i for i in range(self.nodes_num)]

        # Calculate the gap
        tsp_solver = TSPEvaluator(np_points)
        ref_cost = tsp_solver.evaluate(np_ref_tour)
        all_solved_costs = [
            tsp_solver.evaluate(solved_tours[i]) for i in range(self.parallel_sampling)
        ]
        best_solved_cost = np.min(all_solved_costs)
        gap = (best_solved_cost - ref_cost) / ref_cost * 100

        # record the better/worse/match
        better = 0.0
        match = 0.0
        worse = 0.0
        if gap < -1e-12:
            better = 1.0
        elif gap < 1e-12:
            match = 1.0
        else:
            worse = 1.0
        self.gap_list.append(gap)

        # Log the loss and gap
        metrics = {
            f"{phase}/loss": loss,
            f"{phase}/gap": gap,
            f"{phase}/better": better,
            f"{phase}/match": match,
            f"{phase}/worse": worse,
        }

        if phase == "test":
            metrics.update(
                {"test/ref_cost": ref_cost, "test/solved_cost": best_solved_cost}
            )

        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        return metrics

    def solve(
        self,
        data: Union[np.ndarray, torch.Tensor],
        edge_index: Union[np.ndarray, torch.Tensor] = None,
        batch_size: int = 16,
        device="cpu",
    ):
        """solve function, return heatmap"""
        raise NotImplementedError

    def load_weights(self, pretrained_path: str = None):
        raise NotImplementedError

    def loss_fn(self, x_pred, adj_matrix):
        raise NotImplementedError
