from .base import OutLayerBase
from .node import NodeOutLayer
from .edge import EdgeOutLayer


OUTLAYER_DICT = {
    "ATSP": EdgeOutLayer,
    "CVRP": EdgeOutLayer,
    "MCl": NodeOutLayer,
    "MIS": NodeOutLayer,
    "MCut": NodeOutLayer,
    "MVC": NodeOutLayer,
    "TSP": EdgeOutLayer
}

def get_out_layer_by_task(task: str):
    return OUTLAYER_DICT[task]