import re
import time
import json
import subprocess
import numpy as np
import networkx as nx
import os.path
import pathlib
from pathlib import Path


class KaMIS:
    def __init__(
        self,
        weighted: bool=False,
        time_limit: float=600.0,
    ):
        self.weighted = weighted
        self.time_limit = time_limit
        self.kamis_path = pathlib.Path(__file__).parent
    
    @staticmethod
    def __prepare_graph(g: nx.Graph, weighted = False):
        g.remove_edges_from(nx.selfloop_edges(g))
        n = g.number_of_nodes()
        m = g.number_of_edges()
        wt = 0 if not weighted else 10
        res = f"{n} {m} {wt}\n"
        for n, nbrsdict in g.adjacency():
            line = []
            if weighted:
                line.append(g.nodes(data="weight", default=1)[n])
            for nbr, _ in sorted(nbrsdict.items()):
                line.append(nbr + 1)
            res += " ".join(map(str, line)) + "\n"
        return res
    
    @classmethod
    def _prepare_instances(C, instance_directory: pathlib.Path, 
                           cache_directory: pathlib.Path, **kwargs):
        instance_directory = Path(instance_directory)
        cache_directory = Path(cache_directory)
        for graph_path in instance_directory.rglob("*.gpickle"):
            C._prepare_instance(graph_path.resolve(), cache_directory, **kwargs)
            
    def _prepare_instance(
        source_instance_file: pathlib.Path, 
        cache_directory: pathlib.Path, 
        weighted=False
    ):
        cache_directory.mkdir(parents=True, exist_ok=True)
        dest_path = cache_directory / (source_instance_file.stem + \
            f"_{'weighted' if weighted else 'unweighted'}.graph")
        if os.path.exists(dest_path):
            source_mtime = os.path.getmtime(source_instance_file)
            last_updated = os.path.getmtime(dest_path)
            if source_mtime <= last_updated:
                return
        print(f"Updated graph file: {source_instance_file}.")
        g = nx.read_gpickle(source_instance_file)
        graph = KaMIS.__prepare_graph(g, weighted=weighted)
        with open(dest_path, "w") as res_file:
            res_file.write(graph)

    def solve(self, solve_data_path: pathlib.Path, results_path: pathlib.Path):
        print("Solving all given instances using " + str(self))
        cache_directory = os.path.join(solve_data_path, "preprocessed")
        self._prepare_instances(solve_data_path, cache_directory, weighted=self.weighted)
        results = {}
        solve_data_path = Path(solve_data_path)
        results_path = Path(results_path)
        for graph_path in solve_data_path.rglob("*.gpickle"):
            if self.weighted:
                executable = self.kamis_path / "KaMIS" / "deploy" / "weighted_branch_reduce"
            else:
                executable = self.kamis_path / "KaMIS" / "deploy" / "redumis"
            
            _preprocessed_graph = os.path.join(cache_directory, (graph_path.stem + \
                f"_{'weighted' if self.weighted else 'unweighted'}.graph"))
            results_filename = os.path.join(results_path, (graph_path.stem + \
                f"_{'weighted' if self.weighted else 'unweighted'}.result"))
            arguments = [
                _preprocessed_graph, # input
                "--output", results_filename, # output
                "--time_limit", str(self.time_limit),
            ]

            print(f"Calling {executable} with arguments {arguments}.")
            start_time = time.monotonic()
            result = subprocess.run(
                [executable] + arguments, 
                shell=False, 
                capture_output=True, 
                text=True
            )
            lines = result.stdout.split('\n')
            solve_time = time.monotonic() - start_time
            
            results[graph_path.stem] = {"total_time": solve_time}
            with open(results_filename, "r") as f:
                vertices = list(map(int, f.read().replace('\n','')))
            is_vertices = np.flatnonzero(np.array(vertices))

            if self.weighted:
                discovery = re.compile("^(\d+(\.\d*)?) \[(\d+\.\d*)\]$")
                max_mwis_weight = 0
                mis_time = 0.0
                for line in lines:
                    match = discovery.match(line)
                    if match:
                        mwis_weight = float(match[1])
                        if mwis_weight > max_mwis_weight:
                            max_mwis_weight = mwis_weight
                            mis_time = float(match[3])

                if max_mwis_weight == 0:
                    # try another method
                    for line in lines:
                        if line.startswith("time"):
                            mis_time = line.split(" ")[1]
                        if line.startswith("MIS_weight"):
                            max_mwis_weight = line.split(" ")[1]

                if max_mwis_weight == 0:
                    results[graph_path.stem]["mwis_found"] = False
                else:
                    results[graph_path.stem]["mwis_found"] = True
                    results[graph_path.stem]["mwis"] = is_vertices.tolist()
                    results[graph_path.stem]["time_to_find_mwis"] = mis_time
                    results[graph_path.stem]["mwis_vertices"] = is_vertices.shape[0]
                    results[graph_path.stem]["mwis_weight"] = max_mwis_weight

            else:
                stdout = "\n".join(lines)
                discovery = re.compile("Best solution:\s+(\d+)\nTime:\s+(\d+\.\d*)\n", re.MULTILINE)
                time_found_in_stdout = False
                solution_time = 0.0
                for size, timestamp in discovery.findall(stdout):
                    if int(size) == is_vertices.shape[0]:
                        solution_time = float(timestamp)
                        time_found_in_stdout = True
                        break

                if not time_found_in_stdout:
                    # try another regex
                    discovery = re.compile("Best\n={42}\nSize:\s+\d+\nTime found:\s+(\d+\.\d*)", re.MULTILINE)
                    m = discovery.search(stdout)
                    if m:
                        solution_time = float(m.group(1))
                        time_found_in_stdout = True

                if not time_found_in_stdout:
                    results[graph_path.stem]["found_mis"] = False
                else:
                    results[graph_path.stem]["found_mis"] = True
                    results[graph_path.stem]["mis"] = is_vertices.tolist()
                    results[graph_path.stem]["vertices"] = is_vertices.shape[0]
                    results[graph_path.stem]["solution_time"] = solution_time

            with open(results_path / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, sort_keys = True, indent=4)

    def __str__(self) -> str:
        return "kamis"