/**
 * Local construction (degree-based) heuristics for MIS, MVC, MCl, and MCut.
 * Sparse adjacency lists + incremental degrees (MIS/MVC/MCl); incremental
 * column sums for MCut. Same semantics as the Python reference implementations.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr float kMisMvcMaskedDeg = 1.0e6f;
constexpr float kMclMaskedDeg = -1.0f;
constexpr float kMcutMaskedDeg = -1.0f;

inline bool is_edge_binary(float v) { return v > 0.5f; }

/** Build undirected adjacency from upper triangle of dense binary adjacency (diagonal ignored). */
void build_neigh_undirected(const float *adj_in, int n, std::vector<std::vector<int>> *neigh_out) {
    auto &neigh = *neigh_out;
    neigh.assign(n, {});
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (is_edge_binary(adj_in[i * n + j])) {
                neigh[i].push_back(j);
                neigh[j].push_back(i);
            }
        }
    }
}

/** After masking nodes in R, remove incident edges and adjust incremental degrees. */
void remove_nodes_from_graph(const std::vector<int> &R, std::vector<char> *inR_buf,
                             std::vector<std::vector<int>> *neigh, std::vector<float> *deg,
                             const float *nw) {
    auto &inR = *inR_buf;
    auto &neigh_ref = *neigh;
    auto &deg_ref = *deg;
    std::fill(inR.begin(), inR.end(), 0);
    for (int r : R) {
        inR[r] = 1;
    }
    for (int r : R) {
        for (int u : neigh_ref[r]) {
            if (!inR[static_cast<size_t>(u)]) {
                deg_ref[static_cast<size_t>(u)] -= nw[r];
                auto &nu = neigh_ref[u];
                nu.erase(std::remove(nu.begin(), nu.end(), r), nu.end());
            }
        }
        neigh_ref[r].clear();
    }
}

void mis_lc_degree_impl(const float *adj_in, const float *nw, int n, std::int32_t *sol_out) {
    std::vector<std::vector<int>> neigh;
    build_neigh_undirected(adj_in, n, &neigh);
    std::vector<float> deg(n);
    std::vector<char> mask(n, 0);
    std::vector<char> inR(n, 0);
    std::vector<int> R;
    R.reserve(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        float s = 0.f;
        for (int j : neigh[static_cast<size_t>(i)]) {
            s += nw[j];
        }
        deg[static_cast<size_t>(i)] = s - nw[i];
    }
    std::fill(sol_out, sol_out + n, 0);

    while (true) {
        int next_node = -1;
        float min_d = 0.f;
        for (int i = 0; i < n; ++i) {
            if (!mask[static_cast<size_t>(i)]) {
                if (next_node < 0 || deg[static_cast<size_t>(i)] < min_d) {
                    min_d = deg[static_cast<size_t>(i)];
                    next_node = i;
                }
            }
        }
        if (next_node < 0) {
            break;
        }

        const auto &conn = neigh[static_cast<size_t>(next_node)];
        sol_out[next_node] = 1;
        for (int j : conn) {
            sol_out[j] = 0;
        }
        for (int j : conn) {
            mask[static_cast<size_t>(j)] = 1;
        }
        mask[static_cast<size_t>(next_node)] = 1;

        R.clear();
        R.push_back(next_node);
        for (int j : conn) {
            R.push_back(j);
        }
        remove_nodes_from_graph(R, &inR, &neigh, &deg, nw);
    }
}

void mvc_lc_degree_impl(const float *adj_in, const float *nw, int n, std::int32_t *sol_out) {
    std::vector<std::vector<int>> neigh;
    build_neigh_undirected(adj_in, n, &neigh);
    std::vector<float> deg(n);
    std::vector<char> mask(n, 0);
    std::vector<char> inR(n, 0);
    std::vector<int> R;
    R.reserve(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        float s = 0.f;
        for (int j : neigh[static_cast<size_t>(i)]) {
            s += nw[j];
        }
        deg[static_cast<size_t>(i)] = s - nw[i];
    }
    std::fill(sol_out, sol_out + n, 0);

    while (true) {
        int next_node = -1;
        float min_d = 0.f;
        for (int i = 0; i < n; ++i) {
            if (!mask[static_cast<size_t>(i)]) {
                if (next_node < 0 || deg[static_cast<size_t>(i)] < min_d) {
                    min_d = deg[static_cast<size_t>(i)];
                    next_node = i;
                }
            }
        }
        if (next_node < 0) {
            break;
        }

        const auto &conn = neigh[static_cast<size_t>(next_node)];
        for (int j : conn) {
            sol_out[j] = 1;
        }
        sol_out[next_node] = 0;
        for (int j : conn) {
            mask[static_cast<size_t>(j)] = 1;
        }
        mask[static_cast<size_t>(next_node)] = 1;

        R.clear();
        R.push_back(next_node);
        for (int j : conn) {
            R.push_back(j);
        }
        remove_nodes_from_graph(R, &inR, &neigh, &deg, nw);
    }
}

void mcl_lc_degree_impl(const float *adj_in, const float *nw, int n, std::int32_t *sol_out) {
    std::vector<std::vector<int>> neigh(n);
    for (int i = 0; i < n; ++i) {
        neigh[static_cast<size_t>(i)].push_back(i);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (is_edge_binary(adj_in[i * n + j])) {
                neigh[static_cast<size_t>(i)].push_back(j);
                neigh[static_cast<size_t>(j)].push_back(i);
            }
        }
    }
    std::vector<float> deg(n);
    std::vector<char> mask(n, 0);
    std::vector<char> inR(n, 0);
    std::vector<char> is_neigh(n, 0);
    std::vector<int> R;
    R.reserve(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        float s = 0.f;
        for (int j : neigh[static_cast<size_t>(i)]) {
            s += nw[j];
        }
        deg[static_cast<size_t>(i)] = s;
    }
    std::fill(sol_out, sol_out + n, 0);

    while (true) {
        int next_node = -1;
        float max_d = 0.f;
        for (int i = 0; i < n; ++i) {
            if (!mask[static_cast<size_t>(i)]) {
                if (next_node < 0 || deg[static_cast<size_t>(i)] > max_d) {
                    max_d = deg[static_cast<size_t>(i)];
                    next_node = i;
                }
            }
        }
        if (next_node < 0) {
            break;
        }

        std::fill(is_neigh.begin(), is_neigh.end(), 0);
        for (int j : neigh[static_cast<size_t>(next_node)]) {
            is_neigh[static_cast<size_t>(j)] = 1;
        }
        R.clear();
        for (int j = 0; j < n; ++j) {
            if (!is_neigh[static_cast<size_t>(j)]) {
                R.push_back(j);
            }
        }
        for (int j : R) {
            sol_out[j] = 0;
        }
        sol_out[next_node] = 1;
        for (int j : R) {
            mask[static_cast<size_t>(j)] = 1;
        }
        mask[static_cast<size_t>(next_node)] = 1;

        // Match Python: only remove unconnect nodes from the graph; next_node stays in adjacency.
        remove_nodes_from_graph(R, &inR, &neigh, &deg, nw);
    }
}

void mcut_lc_degree_impl(const float *lc_graph_in, int n, std::int32_t *sol_set_a) {
    std::vector<float> degree_a(n);
    std::vector<float> degree_b(n);
    std::vector<char> mask(n, 0);
    std::vector<char> set_a(n, 0);
    std::vector<char> set_b(n, 0);

    for (int j = 0; j < n; ++j) {
        degree_a[static_cast<size_t>(j)] = lc_graph_in[j];
    }
    std::fill(degree_b.begin(), degree_b.end(), 0.f);

    set_a[0] = 1;
    mask[0] = 1;

    while (true) {
        bool all_m = true;
        for (int i = 0; i < n; ++i) {
            if (!mask[static_cast<size_t>(i)]) {
                all_m = false;
                break;
            }
        }
        if (all_m) {
            break;
        }

        for (int j = 0; j < n; ++j) {
            if (mask[static_cast<size_t>(j)]) {
                degree_a[static_cast<size_t>(j)] = kMcutMaskedDeg;
                degree_b[static_cast<size_t>(j)] = kMcutMaskedDeg;
            }
        }

        float max_a = degree_a[0];
        float max_b = degree_b[0];
        for (int j = 1; j < n; ++j) {
            if (degree_a[static_cast<size_t>(j)] > max_a) {
                max_a = degree_a[static_cast<size_t>(j)];
            }
            if (degree_b[static_cast<size_t>(j)] > max_b) {
                max_b = degree_b[static_cast<size_t>(j)];
            }
        }

        if (max_a > max_b) {
            float mv = degree_a[0];
            int next_node = 0;
            for (int j = 1; j < n; ++j) {
                if (degree_a[static_cast<size_t>(j)] > mv) {
                    mv = degree_a[static_cast<size_t>(j)];
                    next_node = j;
                }
            }
            set_b[static_cast<size_t>(next_node)] = 1;
            mask[static_cast<size_t>(next_node)] = 1;
            const float *row = lc_graph_in + next_node * n;
            for (int j = 0; j < n; ++j) {
                degree_b[static_cast<size_t>(j)] += row[j];
            }
        } else {
            float mv = degree_b[0];
            int next_node = 0;
            for (int j = 1; j < n; ++j) {
                if (degree_b[static_cast<size_t>(j)] > mv) {
                    mv = degree_b[static_cast<size_t>(j)];
                    next_node = j;
                }
            }
            set_a[static_cast<size_t>(next_node)] = 1;
            mask[static_cast<size_t>(next_node)] = 1;
            const float *row = lc_graph_in + next_node * n;
            for (int j = 0; j < n; ++j) {
                degree_a[static_cast<size_t>(j)] += row[j];
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        sol_set_a[i] = set_a[static_cast<size_t>(i)] ? 1 : 0;
    }
}

void check_adj_nw(py::array_t<float> adj, py::array_t<float> nodes_weight, int &n) {
    auto ba = adj.request();
    auto bn = nodes_weight.request();
    if (ba.ndim != 2 || ba.shape[0] != ba.shape[1]) {
        throw std::invalid_argument("adj must be a square 2D array.");
    }
    n = static_cast<int>(ba.shape[0]);
    if (bn.ndim != 1 || bn.shape[0] != n) {
        throw std::invalid_argument("nodes_weight must have shape (n,) matching adj.");
    }
    if (!(adj.flags() & py::array::c_style) ||
        !(nodes_weight.flags() & py::array::c_style)) {
        throw std::invalid_argument("adj and nodes_weight must be C-contiguous.");
    }
}

void check_adj_sq(py::array_t<float> adj, int &n) {
    auto ba = adj.request();
    if (ba.ndim != 2 || ba.shape[0] != ba.shape[1]) {
        throw std::invalid_argument("adj must be a square 2D array.");
    }
    n = static_cast<int>(ba.shape[0]);
    if (!(adj.flags() & py::array::c_style)) {
        throw std::invalid_argument("adj must be C-contiguous.");
    }
}

}  // namespace

py::array_t<std::int32_t> mis_lc_degree_bind(py::array_t<float> adj,
                                             py::array_t<float> nodes_weight) {
    int n = 0;
    check_adj_nw(adj, nodes_weight, n);
    auto out = py::array_t<std::int32_t>(n);
    py::buffer_info bo = out.request();
    {
        py::gil_scoped_release release;
        mis_lc_degree_impl(adj.data(), nodes_weight.data(), n,
                           static_cast<std::int32_t *>(bo.ptr));
    }
    return out;
}

py::array_t<std::int32_t> mvc_lc_degree_bind(py::array_t<float> adj,
                                             py::array_t<float> nodes_weight) {
    int n = 0;
    check_adj_nw(adj, nodes_weight, n);
    auto out = py::array_t<std::int32_t>(n);
    py::buffer_info bo = out.request();
    {
        py::gil_scoped_release release;
        mvc_lc_degree_impl(adj.data(), nodes_weight.data(), n,
                           static_cast<std::int32_t *>(bo.ptr));
    }
    return out;
}

py::array_t<std::int32_t> mcl_lc_degree_bind(py::array_t<float> adj,
                                             py::array_t<float> nodes_weight) {
    int n = 0;
    check_adj_nw(adj, nodes_weight, n);
    auto out = py::array_t<std::int32_t>(n);
    py::buffer_info bo = out.request();
    {
        py::gil_scoped_release release;
        mcl_lc_degree_impl(adj.data(), nodes_weight.data(), n,
                           static_cast<std::int32_t *>(bo.ptr));
    }
    return out;
}

py::array_t<std::int32_t> mcut_lc_degree_bind(py::array_t<float> adj_weighted) {
    int n = 0;
    check_adj_sq(adj_weighted, n);
    auto out = py::array_t<std::int32_t>(n);
    py::buffer_info bo = out.request();
    {
        py::gil_scoped_release release;
        mcut_lc_degree_impl(adj_weighted.data(), n, static_cast<std::int32_t *>(bo.ptr));
    }
    return out;
}

PYBIND11_MODULE(lc_degree_impl, m) {
    m.doc() = "Pybind11 LC-degree solvers for MIS, MVC, MCl, and MCut.";

    m.def("mis_lc_degree", &mis_lc_degree_bind, py::arg("adj"), py::arg("nodes_weight"),
          "Return int32 solution (0/1) for MIS.");
    m.def("mvc_lc_degree", &mvc_lc_degree_bind, py::arg("adj"), py::arg("nodes_weight"),
          "Return int32 solution (0/1) for MVC.");
    m.def("mcl_lc_degree", &mcl_lc_degree_bind, py::arg("adj"), py::arg("nodes_weight"),
          "Return int32 solution (0/1) for MCl.");
    m.def("mcut_lc_degree", &mcut_lc_degree_bind, py::arg("adj_weighted"),
          "Return int32 membership for side A in max-cut (0/1).");
}
