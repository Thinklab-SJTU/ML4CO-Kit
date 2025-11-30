from .base import OutLayerBase
from .node import NodeOutLayer
from .edge import EdgeOutLayer
from ml4co_kit.task.base import TASK_TYPE


OUTLAYER_DICT = {
<<<<<<< HEAD
    "ATSP": EdgeOutLayer,
    "CVRP": EdgeOutLayer,
    "MCl": NodeOutLayer,
    "MIS": NodeOutLayer,
    "MCut": NodeOutLayer,
    "MVC": NodeOutLayer,
    "TSP": EdgeOutLayer,
=======
    TASK_TYPE.ATSP: EdgeOutLayer,
    TASK_TYPE.TSP: EdgeOutLayer,
    TASK_TYPE.MCL: NodeOutLayer,
    TASK_TYPE.MIS: NodeOutLayer,
    TASK_TYPE.MCUT: NodeOutLayer,
    TASK_TYPE.MVC: NodeOutLayer,
>>>>>>> upstream/main
}


def get_out_layer_by_task(task_type: TASK_TYPE):
    return OUTLAYER_DICT[task_type]