import numpy as np
from enum import Enum
from typing import Union


def to_numpy(x: Union[np.ndarray, list]):
    if type(x) == list:
        return np.array(x)
    elif type(x) == np.ndarray:
        return x
    else:
        raise NotImplementedError()
    

class TASK_TYPE(str, Enum):
    ATSP = "Asymmetric Traveling Salesman Problem (ATSP)"
    CVRP = "Capacitated Vehicle Routing Problem (CVRP)"
    MCl = "Maximum Clique (MCl)"
    MCut = "Maximum Cut (MCut)"
    MIS = "Maximum Independent Set (MIS)"
    MVC = "Minimum Vertex Cover (MVC)"
    TSP = "Traveling Salesman Problem (TSP)"
    LP = "Linear Program (LP)"


class SOLVER_TYPE(str, Enum):
    CONCORDE = "PyConcorde" # Support TSP
    CONCORDE_LARGE = "PyConcorde(Large)" # Support TSP
    GA_EAX = "GA-EAX" # Support TSP
    GA_EAX_LARGE = "GA-EAX(Large)" # Support TSP
    GUROBI = "Gurobi" # Support for MIS, MVC, MC, MCL
    HGS = "HGS" # Support CVRP
    KAMIS = "KaMIS" # Support MIS
    LKH = "LKH" # Support for TSP, ATSP, CVRP
    ML4ATSP = "ML4ATSP" # part of ML4CO
    ML4CVRP = "ML4CVRP" # part of ML4CO
    ML4LP = "ML4LP" # ML4LP
    ML4MCl = "ML4MCl" # part of ML4CO
    ML4MCut = "ML4MCut" # part of ML4CO
    ML4MIS = "ML4MIS" # part of ML4CO
    ML4MVC = "ML4MVC" # part of ML4CO
    ML4TSP = "ML4TSP" # part of ML4CO
    PYVRP = "PyVRP" # Support CVRP


TASK_SUPPORT_SOLVER = {
    TASK_TYPE.ATSP: [SOLVER_TYPE.LKH],
    TASK_TYPE.TSP: [
        SOLVER_TYPE.CONCORDE, SOLVER_TYPE.CONCORDE_LARGE, SOLVER_TYPE.GA_EAX, 
        SOLVER_TYPE.GA_EAX_LARGE, SOLVER_TYPE.LKH
    ],
    TASK_TYPE.CVRP: [SOLVER_TYPE.HGS, SOLVER_TYPE.LKH, SOLVER_TYPE.PYVRP],
    TASK_TYPE.MCl: [SOLVER_TYPE.GUROBI],
    TASK_TYPE.MCut: [SOLVER_TYPE.GUROBI],
    TASK_TYPE.MIS: [SOLVER_TYPE.GUROBI, SOLVER_TYPE.KAMIS],
    TASK_TYPE.MVC: [SOLVER_TYPE.GUROBI],
}
    