import re
import time
import json
import pickle
import shutil
import os.path
import pathlib
import subprocess
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from typing import Union, List
from ml4co_kit.solver.mis.base import MISSolver
from ml4co_kit.utils.graph.mis import MISGraphData
from ml4co_kit.utils.type_utils import SOLVER_TYPE


class KaMISSolver(MISSolver):
    def __init__(
        self,
        weighted: bool = False,
        time_limit: float = 60.0,
    ):
        """
        KaMIS
        Args:
            weighted (bool, optional):
                If enabled, solve the weighted MIS problem instead of MIS.
            time_limit (float, optional):
                Time limit in seconds.
        """
        super(KaMISSolver, self).__init__(
            solver_type=SOLVER_TYPE.KAMIS, weighted=weighted, time_limit=time_limit
        )
        self.kamis_path = pathlib.Path(__file__).parent

    @staticmethod
    def __prepare_graph(g: nx.Graph, weighted=False):
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

    def prepare_instances(
        self, instance_directory: pathlib.Path, cache_directory: pathlib.Path
    ):
        instance_directory = Path(instance_directory)
        cache_directory = Path(cache_directory)
        for graph_path in instance_directory.rglob("*.gpickle"):
            self.prepare_instance(graph_path.resolve(), cache_directory)

    def prepare_instance(
        self,
        source_instance_file: Union[str, pathlib.Path],
        cache_directory: Union[str, pathlib.Path],
    ):
        # check file path
        source_instance_file = Path(source_instance_file)
        cache_directory = Path(cache_directory)
        cache_directory.mkdir(parents=True, exist_ok=True)
        dest_path = cache_directory / (
            source_instance_file.stem
            + f"_{'weighted' if self.weighted else 'unweighted'}.graph"
        )
        if os.path.exists(dest_path):
            source_mtime = os.path.getmtime(source_instance_file)
            last_updated = os.path.getmtime(dest_path)
            if source_mtime <= last_updated:
                return

        # load gpickle data
        with open(source_instance_file, mode="rb") as f:
            g = pickle.load(f)
        
        # prepare graph (core)   
        graph = KaMISSolver.__prepare_graph(g, weighted=self.weighted)
        with open(dest_path, "w") as res_file:
            res_file.write(graph)

    def solve(
        self, src: Union[str, pathlib.Path], out: Union[str, pathlib.Path],
    ) -> List[MISGraphData]:
        message = (
            "Please check KaMIS compilation. " 
            "you can try ``self.recompile_kamis()``. "
            "If you are sure that the ``KaMIS`` is correct, "
            "please confirm whether the Conda environment of the terminal"
            "is consistent with the Python environment."
        )  
        src = Path(src)
        out = Path(out)
        try:
            self._solve(src, out)
        except TypeError:
            raise TypeError(message)
        except FileNotFoundError:
            raise FileNotFoundError(message)
        self.from_gpickle_result_folder(
            gpickle_folder_path=src, result_folder_path=out, ref=False, cover=True
        )
        return self.graph_data

    def _solve(self, src: Union[str, pathlib.Path], out: Union[str, pathlib.Path]):
        src = Path(src)
        out = Path(out)
        if not os.path.exists(out):
            os.makedirs(out)
        cache_directory = src / "preprocessed"
        self.prepare_instances(src, cache_directory)
        results = {}
        src = Path(src)
        out = Path(out)
        files = [f for f in os.listdir(src) if f.endswith(".gpickle")]
        for graph_path in tqdm(files, desc=self.solve_msg):
            graph_path = graph_path.replace(".gpickle", "")
            if self.weighted:
                executable = (
                    self.kamis_path / "KaMIS" / "deploy" / "weighted_branch_reduce"
                )
            else:
                executable = self.kamis_path / "KaMIS" / "deploy" / "redumis"
            _preprocessed_graph = os.path.join(
                cache_directory,
                (
                    graph_path
                    + f"_{'weighted' if self.weighted else 'unweighted'}.graph"
                ),
            )
            results_filename = os.path.join(
                out,
                (
                    graph_path
                    + f"_{'weighted' if self.weighted else 'unweighted'}.result"
                ),
            )
            arguments = [
                _preprocessed_graph,  # input
                "--output",
                results_filename,  # output
                "--time_limit",
                str(self.time_limit),
            ]

            start_time = time.monotonic()
            result = subprocess.run(
                [executable] + arguments, shell=False, capture_output=True, text=True
            )
            lines = result.stdout.split("\n")
            solve_time = time.monotonic() - start_time

            results[graph_path] = {"total_time": solve_time}
            with open(results_filename, "r") as f:
                vertices = list(map(int, f.read().replace("\n", "")))
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
                    results[graph_path]["mwis_found"] = False
                else:
                    results[graph_path]["mwis_found"] = True
                    results[graph_path]["mwis"] = is_vertices.tolist()
                    results[graph_path]["time_to_find_mwis"] = mis_time
                    results[graph_path]["mwis_vertices"] = is_vertices.shape[0]
                    results[graph_path]["mwis_weight"] = max_mwis_weight

            else:
                stdout = "\n".join(lines)
                discovery = re.compile(
                    "Best solution:\s+(\d+)\nTime:\s+(\d+\.\d*)\n", re.MULTILINE
                )
                time_found_in_stdout = False
                solution_time = 0.0
                for size, timestamp in discovery.findall(stdout):
                    if int(size) == is_vertices.shape[0]:
                        solution_time = float(timestamp)
                        time_found_in_stdout = True
                        break

                if not time_found_in_stdout:
                    # try another regex
                    discovery = re.compile(
                        "Best\n={42}\nSize:\s+\d+\nTime found:\s+(\d+\.\d*)",
                        re.MULTILINE,
                    )
                    m = discovery.search(stdout)
                    if m:
                        solution_time = float(m.group(1))
                        time_found_in_stdout = True

                if not time_found_in_stdout:
                    results[graph_path]["found_mis"] = False
                else:
                    results[graph_path]["found_mis"] = True
                    results[graph_path]["mis"] = is_vertices.tolist()
                    results[graph_path]["vertices"] = is_vertices.shape[0]
                    results[graph_path]["solution_time"] = solution_time

            with open(out / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, sort_keys=True, indent=4)

    def recompile_kamis(self):
        if os.path.exists(self.kamis_path / "KaMIS/deploy/"):
            shutil.rmtree(self.kamis_path / "KaMIS/deploy/")
        if os.path.exists(self.kamis_path / "KaMIS/tmp_build/"):
            shutil.rmtree(self.kamis_path / "KaMIS/tmp_build/")
        shutil.copytree(
            self.kamis_path / "kamis-source/", self.kamis_path / "KaMIS/tmp_build/"
        )
        ori_dir = os.getcwd()
        os.chdir(self.kamis_path / "KaMIS/tmp_build/")
        os.system("bash cleanup.sh")
        os.system("bash compile_withcmake.sh")
        os.chdir(ori_dir)
        shutil.copytree(
            self.kamis_path / "KaMIS/tmp_build/deploy/",
            self.kamis_path / "KaMIS/deploy/",
        )
        shutil.rmtree(self.kamis_path / "KaMIS/tmp_build/")

    def __str__(self) -> str:
        return "KaMISSolver"
