from .base import GNN4COEmbedder
from .atsp import ATSPEmbedder
from .mcl import MClEmbedder
from .mcut import MCutEmbedder
from .mis import MISEmbedder
from .mvc import MVCEmbedder
from .tsp import TSPEmbedder


EMBEDDER_DICT = {
    "ATSP": ATSPEmbedder,
    "MCl": MClEmbedder,
    "MIS": MISEmbedder,
    "MCut": MCutEmbedder,
    "MVC": MVCEmbedder,
    "TSP": TSPEmbedder
}

def get_embedder_by_task(task: str):
    return EMBEDDER_DICT[task]