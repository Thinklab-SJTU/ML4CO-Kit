import io
import re
import os
import sys
import time
import json
import copy
import pathlib
import pulp as plp
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from .base import MISSolver


class MISGurobi(MISSolver):
    def __init__(
        self,
        weighted: bool=False,
        time_limit: float=60.0,
        num_threads: int=8,
        quadratic: bool=False,
        write_mps: bool=False,
    ):
        """
        MISGurobi
        Args:
            weighted (bool, optional):
                If enabled, solve the weighted MIS problem instead of MIS.
            time_limit (float, optional): 
                Time limit in seconds. 
            num_threads (int, optional): 
                Maximum number of threads to use.
            quadratic (bool, optional): 
                Whether a quadratic program should be used instead of a linear program 
                to solve the MIS problem (cannot be used together with weighted).
            write_mps (bool, optional): 
                Instead of solving, write mps output (e.g., for tuning)
        """
        super(MISGurobi, self).__init__()
        self.solver_type = "gurobi"
        self.weighted = weighted
        self.time_limit = time_limit
        self.num_threads = num_threads
        self.quadratic = quadratic
        self.write_mps = write_mps
        self.gurobi_path = pathlib.Path(__file__).parent

    @staticmethod
    def __prepare_graph(g: nx.Graph, weighted = False):
        graph = copy.deepcopy(g)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        # the gurobi solver file always expects a weighted file
        # however, for "unweighted" we supply all weights as 1
        if not weighted:
            nx.set_node_attributes(graph, 1, name="weight")
        return graph

    def prepare_instances(self, instance_directory: pathlib.Path, 
                           cache_directory: pathlib.Path):
        for graph_path in instance_directory.rglob("*.gpickle"):
            self.prepare_instance(graph_path.resolve(), cache_directory)

    def prepare_instance(
        self,
        source_instance_file: pathlib.Path, 
        cache_directory: pathlib.Path,
    ):
        source_instance_file = Path(source_instance_file)
        cache_directory = Path(cache_directory)
        cache_directory.mkdir(parents=True, exist_ok=True)

        dest_path = cache_directory / (source_instance_file.stem + \
            f"_{'weighted' if self.weighted else 'unweighted'}.graph")
        
        if os.path.exists(dest_path):
            source_mtime = os.path.getmtime(source_instance_file)
            last_updated = os.path.getmtime(dest_path)
            if source_mtime <= last_updated:
                return # we already have an up2date version of that file

        print(f"Updated graph file: {source_instance_file}.")
        g = nx.read_gpickle(source_instance_file)
        graph = self.__prepare_graph(g, weighted=self.weighted)
        nx.write_gpickle(graph, dest_path)
    
    def solve(self, src: pathlib.Path, out: pathlib.Path):
        print("Solving all given instances using " + str(self))
        # preprocess
        src = Path(src)
        out = Path(out)
        cache_directory = src / "preprocessed"
        self.prepare_instances(src, cache_directory)
        # call gurobi
        results = {}
        for _, graph_path in enumerate(
            cache_directory.rglob(f"*_{'weighted' if self.weighted else 'unweighted'}.graph")
        ):
            graph_name_stub = os.path.splitext(os.path.basename(graph_path))[0].rsplit('_', 1)[0]
            print(f"Solving graph {graph_path}")
            graph = nx.read_gpickle(graph_path)
            if self.write_mps:
                self._solve(graph, graph_name_stub, out)
            else:
                solu, sum, status, total_time, explore_time = \
                    self._solve(graph, graph_name_stub, out)
                print(f"Found MWIS: n={len(solu)}, w={sum} ✔️✔️")
                if status != "Optimal":
                    print(f"Non-Optimal Gurobi status: {status}")
                results[graph_name_stub] = {
                    "total_time": total_time,
                    "mwis_vertices": int(len(solu)),
                    "mwis_weight": float(sum),
                    "mwis": np.ravel(solu).tolist(),
                    "gurobi_status": status,
                    "gurobi_explore_time": explore_time
                }
        # write the result
        if not self.write_mps:
            with open(out / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, sort_keys = True, indent=4)
        print("Done with all graphs, exiting.")

    def _solve(self, graph: nx.Graph, graph_name_stub: str, out: pathlib.Path):
        start_time = time.monotonic()
        weight_dict = nx.get_node_attributes(graph, "weight")
        # not defined whether dictionary returned by networkx is sorted
        weight_list = [weight_dict[key] for key in sorted(weight_dict.keys())]
        wts = np.array(weight_list)
    
        if not self.quadratic:
            self.non_quadratic_solve(
                start_time=start_time,
                graph=graph,
                wts=wts,
                graph_name_stub=graph_name_stub,
                out=out
            )
        else:
            self.quadratic_solve(
                start_time=start_time,
                graph=graph,
                wts=wts,
                graph_name_stub=graph_name_stub,
                out=out
            )    
    
    def quadratic_solve(
        self,
        start_time: float,
        graph: nx.Graph, 
        wts: np.ndarray,
        graph_name_stub: str,
        out: pathlib.Path
    ):
        import gurobipy as gp
        from gurobipy import GRB

        n = graph.number_of_nodes()
        adj = nx.to_numpy_array(graph)
        J = np.identity(n)
        A = J - adj
        m = gp.Model("mis")
        x = m.addMVar(shape=n, vtype=GRB.BINARY, name="x")
        m.setObjective(x @ A @ x, GRB.MAXIMIZE)
        m.setParam('TimeLimit', self.time_limit)
        m.setParam('Threads', self.num_threads)
        m.setParam('ImproveStartTime', self.time_limit * 0.9)

        if self.write_mps:
            m.write(out / f"{graph_name_stub}.mps")
            return

        # redirect stdout by gurobi into internal string buffer
        real_stdout = sys.stdout
        sys.stdout = io.StringIO()
        m.optimize()
        gurobi_output = sys.stdout.getvalue()
        sys.stdout = real_stdout # set stdout file back to normal
        print(gurobi_output)
        # parse solving time from gurobi output
        time_string_match = re.compile("in (\d+(\.\d*)) seconds").findall(gurobi_output)[0]
        apparent_solve_time = float(time_string_match[0])
        solu = np.nonzero(np.array(x.X))[0]
        
        # status
        status = m.status
        if status == GRB.OPTIMAL:
            status = "Optimal"
        else:
            status = str(status)
        end_time = time.monotonic()
        return solu, wts[solu].sum(), status, end_time - start_time, apparent_solve_time
    
    def non_quadratic_solve(
        self,
        start_time: float,
        graph: nx.Graph, 
        wts: np.ndarray,
        graph_name_stub: str,
        out: pathlib.Path
    ):
        opt_model = plp.LpProblem(name="model")
        adj = nx.adjacency_matrix(graph)
        x_vars = {i: plp.LpVariable(cat=plp.LpBinary, name="x_{0}".format(i)) for i in range(wts.size)}
        # not sure why we need set here, as the call to range produces a unique list
        set_V = set(range(wts.size)) 
        constraints = dict()
        ei = 0 # number of constraints counter
        
        # we iterate over all nodes (could we be more explicit and use G.nodes here?)
        for j in set_V:
            # set N are all the vertices that j has a connection to
            _, set_N = np.nonzero(adj[j]) 
            for i in set_N:
                constraints[ei] = opt_model.addConstraint(
                    plp.LpConstraint(
                        e=plp.lpSum([x_vars[i], x_vars[j]]),
                        sense=plp.LpConstraintLE,
                        rhs=1,
                        # constraint that we only pick one of two neighboring nodes
                        name="constraint_{0}_{1}".format(j,i)))
                ei = ei + 1
        
        # we want to maximize the weight
        objective = plp.lpSum(x_vars[i] * wts[i] for i in set_V)
        opt_model.sense = plp.LpMaximize
        opt_model.setObjective(objective)

        if self.write_mps:
            opt_model.writeMPS(out / (graph_name_stub  + ".mps"))
            return

        # redirect stdout by gurobi into internal string buffer
        real_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # solve
        opt_model.solve(
            solver=plp.apis.GUROBI(
                mip=True,
                timeLimit=self.time_limit,
                Threads=self.num_threads,
                ImproveStartTime = self.time_limit * 0.9
            )
        )
    
        gurobi_output = sys.stdout.getvalue()
        sys.stdout = real_stdout # set stdout file back to normal
        print(gurobi_output)

        # parse solving time from gurobi output
        time_string_match = re.compile("in (\d+(\.\d*)) seconds").findall(gurobi_output)[0]
        apparent_solve_time = float(time_string_match[0])

        if plp.LpStatus[opt_model.status] == "Optimal":
            opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns=["variable_object"])
            opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)
            solu = opt_df[opt_df['solution_value'] > 0].index.to_numpy()
        else:
            # Retrieve best (not optimal) solution
            # GUROBI specific, seems to be a bug in PuLP where varValue is None for non-optimal solutions
            solu = np.nonzero(np.array(opt_model.solverModel.X))[0]
        
        end_time = time.monotonic()
        return solu, wts[solu].sum(), plp.LpStatus[opt_model.status], \
            end_time-start_time, apparent_solve_time
    
    
    def __str__(self) -> str:
        return "gurobi"