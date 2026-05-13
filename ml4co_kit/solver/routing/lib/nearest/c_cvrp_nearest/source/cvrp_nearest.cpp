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

py::array_t<int> cvrp_nearest_segment(
    py::array_t<float, py::array::c_style | py::array::forcecast> depot,
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    py::array_t<float, py::array::c_style | py::array::forcecast> norm_demands
) {
    auto dep = depot.request();
    auto pts = points.request();
    auto dem = norm_demands.request();

    if (dep.ndim != 1) {
        throw std::runtime_error("depot must be 1D with shape (dim,).");
    }
    if (pts.ndim != 2) {
        throw std::runtime_error("points must be 2D with shape (N, dim).");
    }
    if (dem.ndim != 1) {
        throw std::runtime_error("norm_demands must be 1D with shape (N,).");
    }

    const int n = static_cast<int>(pts.shape[0]);
    const int dim = static_cast<int>(pts.shape[1]);
    if (static_cast<int>(dem.shape[0]) != n) {
        throw std::runtime_error("points and norm_demands size mismatch.");
    }
    if (static_cast<int>(dep.shape[0]) != dim) {
        throw std::runtime_error("depot dim and points dim mismatch.");
    }

    const auto* d0 = static_cast<float*>(dep.ptr);
    const auto* p = static_cast<float*>(pts.ptr);
    const auto* q = static_cast<float*>(dem.ptr);

    std::vector<unsigned char> visited(n, 0);
    std::vector<int> route;
    route.reserve(n + 2);
    route.push_back(0);

    float remain = 1.0f;
    std::vector<float> cur(dim);
    for (int k = 0; k < dim; ++k) cur[k] = d0[k];

    while (true) {
        int best = -1;
        float best_d = std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; ++i) {
            if (visited[i]) continue;
            if (q[i] > remain + 1e-7f) continue;
            const float* pi = p + i * dim;
            const float d = sq_dist(cur.data(), pi, dim);
            if (d < best_d) {
                best_d = d;
                best = i;
            }
        }

        if (best < 0) break;

        visited[best] = 1;
        remain -= q[best];
        route.push_back(best + 1);
        const float* pb = p + best * dim;
        for (int k = 0; k < dim; ++k) cur[k] = pb[k];
    }

    route.push_back(0);

    py::array_t<int> out(route.size());
    auto out_m = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < static_cast<ssize_t>(route.size()); ++i) {
        out_m(i) = route[i];
    }
    return out;
}

PYBIND11_MODULE(cvrp_nearest_impl, m) {
    m.doc() = "PyBind11 nearest-neighbor implementation for CVRP segment";
    m.def("cvrp_nearest_segment", &cvrp_nearest_segment, "Nearest feasible CVRP segment");
}
