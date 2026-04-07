#include <vector>
#include <limits>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

inline float sq_dist(const float* a, const float* b, int dim) {
    float v = 0.0f;
    for (int k = 0; k < dim; ++k) {
        const float d = a[k] - b[k];
        v += d * d;
    }
    return v;
}

py::array_t<int> tsp_nearest(py::array_t<float, py::array::c_style | py::array::forcecast> points) {
    auto buf = points.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("points must be 2D with shape (N, dim).");
    }

    const int n = static_cast<int>(buf.shape[0]);
    const int dim = static_cast<int>(buf.shape[1]);
    if (n <= 0) {
        throw std::runtime_error("points must be non-empty.");
    }

    const auto* p = static_cast<float*>(buf.ptr);
    std::vector<unsigned char> visited(n, 0);
    std::vector<int> route;
    route.reserve(n + 1);

    int cur = 0;
    visited[cur] = 1;
    route.push_back(cur);

    for (int step = 1; step < n; ++step) {
        int best = -1;
        float best_d = std::numeric_limits<float>::infinity();
        const float* cur_ptr = p + cur * dim;

        for (int j = 0; j < n; ++j) {
            if (visited[j]) continue;
            const float* nxt_ptr = p + j * dim;
            const float d = sq_dist(cur_ptr, nxt_ptr, dim);
            if (d < best_d) {
                best_d = d;
                best = j;
            }
        }

        if (best < 0) {
            throw std::runtime_error("failed to find next unvisited node in tsp_nearest.");
        }
        visited[best] = 1;
        route.push_back(best);
        cur = best;
    }

    route.push_back(0);

    py::array_t<int> out(route.size());
    auto out_m = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < static_cast<ssize_t>(route.size()); ++i) {
        out_m(i) = route[i];
    }
    return out;
}

PYBIND11_MODULE(tsp_nearest_impl, m) {
    m.doc() = "PyBind11 nearest-neighbor implementation for TSP";
    m.def("tsp_nearest", &tsp_nearest, "Nearest-neighbor TSP route from node 0");
}
