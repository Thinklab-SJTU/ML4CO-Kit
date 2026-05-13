#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct Rect {
    double xl;
    double yl;
    double xh;
    double yh;
};

py::array_t<double, py::array::c_style | py::array::forcecast> as_double_array(
    const py::array& array,
    const char* name
) {
    py::array_t<double, py::array::c_style | py::array::forcecast> casted(array);
    if (!casted) {
        throw std::runtime_error(std::string("failed to cast ") + name + " to float array");
    }
    return casted;
}

py::array_t<bool, py::array::c_style | py::array::forcecast> as_bool_array(
    const py::object& object,
    const char* name
) {
    py::array_t<bool, py::array::c_style | py::array::forcecast> casted(object);
    if (!casted) {
        throw std::runtime_error(std::string("failed to cast ") + name + " to bool array");
    }
    return casted;
}

void check_2d_width(const py::buffer_info& info, const char* name, ssize_t width) {
    if (info.ndim != 2 || info.shape[1] != width) {
        throw std::runtime_error(std::string(name) + " should be a 2D array with shape (N, " + std::to_string(width) + ").");
    }
}

void check_1d_width(const py::buffer_info& info, const char* name, ssize_t width) {
    if (info.ndim != 1 || info.shape[0] != width) {
        throw std::runtime_error(std::string(name) + " should be a 1D array with shape (" + std::to_string(width) + ",).");
    }
}

int clamp_bin(double value, double origin, double bin_size, int bins) {
    if (bins <= 1) {
        return 0;
    }
    const int idx = static_cast<int>(std::floor((value - origin) / bin_size));
    return std::min(std::max(idx, 0), bins - 1);
}

double rect_overlap_area(const Rect& a, const Rect& b) {
    const double overlap_w = std::min(a.xh, b.xh) - std::max(a.xl, b.xl);
    if (overlap_w <= 0.0) {
        return 0.0;
    }
    const double overlap_h = std::min(a.yh, b.yh) - std::max(a.yl, b.yl);
    if (overlap_h <= 0.0) {
        return 0.0;
    }
    return overlap_w * overlap_h;
}

std::vector<Rect> read_die_rects(const py::buffer_info& die_info) {
    const double* die_ptr = static_cast<const double*>(die_info.ptr);
    std::vector<Rect> die_rects;

    if (die_info.ndim == 1 && die_info.shape[0] == 4) {
        die_rects.push_back(Rect{die_ptr[0], die_ptr[1], die_ptr[2], die_ptr[3]});
    } else if (die_info.ndim == 2 && die_info.shape[1] == 4) {
        die_rects.reserve(static_cast<size_t>(die_info.shape[0]));
        for (ssize_t i = 0; i < die_info.shape[0]; ++i) {
            die_rects.push_back(Rect{
                die_ptr[i * 4],
                die_ptr[i * 4 + 1],
                die_ptr[i * 4 + 2],
                die_ptr[i * 4 + 3]
            });
        }
    } else {
        throw std::runtime_error("die should be a 1D array of shape (4,) or a 2D array of shape (R, 4).");
    }

    if (die_rects.empty()) {
        throw std::runtime_error("die should contain at least one rectangle.");
    }
    for (const Rect& rect : die_rects) {
        if (rect.xh <= rect.xl || rect.yh <= rect.yl) {
            throw std::runtime_error("each die rectangle should satisfy xh > xl and yh > yl.");
        }
    }
    return die_rects;
}

Rect rects_bbox(const std::vector<Rect>& rects) {
    Rect bbox{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()
    };
    for (const Rect& rect : rects) {
        bbox.xl = std::min(bbox.xl, rect.xl);
        bbox.yl = std::min(bbox.yl, rect.yl);
        bbox.xh = std::max(bbox.xh, rect.xh);
        bbox.yh = std::max(bbox.yh, rect.yh);
    }
    return bbox;
}

bool rect_inside_any(const Rect& rect, const std::vector<Rect>& regions) {
    for (const Rect& region : regions) {
        if (rect.xl >= region.xl && rect.yl >= region.yl &&
            rect.xh <= region.xh && rect.yh <= region.yh) {
            return true;
        }
    }
    return false;
}

std::pair<int, int> resolve_grid(int bin_cols, int bin_rows) {
    if (bin_cols <= 0 || bin_rows <= 0) {
        throw std::runtime_error("bin_cols and bin_rows should be positive.");
    }
    return {bin_cols, bin_rows};
}

}  // namespace

class EDAHelper {
public:
    py::tuple check_constraints(
        const py::array& sol_array,
        const py::array& die_array,
        const py::array& cells_array,
        const py::object& macro_mask_object = py::none()
    ) const {
        auto sol = as_double_array(sol_array, "sol");
        auto die = as_double_array(die_array, "die");
        auto cells = as_double_array(cells_array, "cells");

        const auto sol_info = sol.request();
        const auto die_info = die.request();
        const auto cells_info = cells.request();
        check_2d_width(sol_info, "sol", 2);
        check_2d_width(cells_info, "cells", 2);
        if (sol_info.shape[0] != cells_info.shape[0]) {
            throw std::runtime_error("sol and cells should have the same number of rows.");
        }

        const ssize_t n = sol_info.shape[0];
        const double* sol_ptr = static_cast<const double*>(sol_info.ptr);
        const double* cells_ptr = static_cast<const double*>(cells_info.ptr);
        const std::vector<Rect> die_rects = read_die_rects(die_info);
        std::vector<bool> skip_boundary(static_cast<size_t>(n), false);
        if (!macro_mask_object.is_none()) {
            auto macro_mask = as_bool_array(macro_mask_object, "macro_mask");
            const auto macro_mask_info = macro_mask.request();
            check_1d_width(macro_mask_info, "macro_mask", n);
            const bool* macro_mask_ptr = static_cast<const bool*>(macro_mask_info.ptr);
            for (ssize_t i = 0; i < n; ++i) {
                skip_boundary[static_cast<size_t>(i)] = macro_mask_ptr[i];
            }
        }

        bool inside_die = true;
        std::vector<Rect> rects(static_cast<size_t>(n));
        std::vector<int> boundary_violations;
        for (ssize_t i = 0; i < n; ++i) {
            const double x = sol_ptr[i * 2];
            const double y = sol_ptr[i * 2 + 1];
            const double w = cells_ptr[i * 2];
            const double h = cells_ptr[i * 2 + 1];
            if (w < 0.0 || h < 0.0) {
                throw std::runtime_error("cell width and height should be non-negative.");
            }
            rects[static_cast<size_t>(i)] = Rect{x, y, x + w, y + h};
            if (!skip_boundary[static_cast<size_t>(i)] && !rect_inside_any(rects[static_cast<size_t>(i)], die_rects)) {
                inside_die = false;
                boundary_violations.push_back(static_cast<int>(i));
            }
        }
        if (!boundary_violations.empty()) {
            std::cout << "[EDAHelper.check_constraints] inside_die=false; "
                      << boundary_violations.size()
                      << " non-macro cells are outside the placement die rectangles."
                      << std::endl;
            for (const int idx : boundary_violations) {
                const Rect& rect = rects[static_cast<size_t>(idx)];
                std::cout << "  cell_idx=" << idx
                          << " rect=(" << rect.xl << ", " << rect.yl
                          << ", " << rect.xh << ", " << rect.yh << ")"
                          << " size=(" << rect.xh - rect.xl
                          << ", " << rect.yh - rect.yl << ")"
                          << std::endl;
            }
        }

        std::vector<int> order;
        order.reserve(static_cast<size_t>(n));
        for (ssize_t i = 0; i < n; ++i) {
            const Rect& r = rects[static_cast<size_t>(i)];
            if (r.xh > r.xl && r.yh > r.yl) {
                order.push_back(static_cast<int>(i));
            }
        }
        std::sort(order.begin(), order.end(), [&rects](int a, int b) {
            const Rect& ra = rects[static_cast<size_t>(a)];
            const Rect& rb = rects[static_cast<size_t>(b)];
            if (ra.xl == rb.xl) {
                return ra.xh < rb.xh;
            }
            return ra.xl < rb.xl;
        });

        long double overlap_sum = 0.0L;
        std::vector<int> active;
        active.reserve(1024);
        for (const int idx : order) {
            const Rect& current = rects[static_cast<size_t>(idx)];
            size_t write_pos = 0;
            for (const int active_idx : active) {
                const Rect& candidate = rects[static_cast<size_t>(active_idx)];
                if (candidate.xh > current.xl) {
                    active[write_pos++] = active_idx;
                    if (candidate.yh > current.yl && current.yh > candidate.yl) {
                        overlap_sum += rect_overlap_area(candidate, current);
                    }
                }
            }
            active.resize(write_pos);
            active.push_back(idx);
        }

        return py::make_tuple(inside_die, static_cast<long long>(std::llround(overlap_sum)));
    }

    py::tuple evaluate(
        const py::array& sol_array,
        const py::iterable& nets,
        const py::array& die_array,
        int bin_cols = 224,
        int bin_rows = 224
    ) const {
        auto sol = as_double_array(sol_array, "sol");
        auto die = as_double_array(die_array, "die");

        const auto sol_info = sol.request();
        const auto die_info = die.request();
        check_2d_width(sol_info, "sol", 2);

        const ssize_t cell_num = sol_info.shape[0];
        const double* sol_ptr = static_cast<const double*>(sol_info.ptr);
        const std::vector<Rect> die_rects = read_die_rects(die_info);
        const Rect die_bbox = rects_bbox(die_rects);
        const double die_xl = die_bbox.xl;
        const double die_yl = die_bbox.yl;
        const double die_xh = die_bbox.xh;
        const double die_yh = die_bbox.yh;
        const double die_w = die_xh - die_xl;
        const double die_h = die_yh - die_yl;
        if (die_w <= 0.0 || die_h <= 0.0) {
            throw std::runtime_error("die should satisfy xh > xl and yh > yl.");
        }

        const auto [cols, rows] = resolve_grid(bin_cols, bin_rows);
        const double bin_w = die_w / static_cast<double>(cols);
        const double bin_h = die_h / static_cast<double>(rows);
        const double bin_area = bin_w * bin_h;
        std::vector<double> rudy_map(static_cast<size_t>(cols) * rows, 0.0);

        long double hpwl = 0.0L;
        for (const py::handle& net_handle : nets) {
            py::array net_obj = py::reinterpret_borrow<py::array>(net_handle);
            auto net = as_double_array(net_obj, "net");
            const auto net_info = net.request();
            check_2d_width(net_info, "net", 3);
            const ssize_t pin_num = net_info.shape[0];
            if (pin_num <= 1) {
                continue;
            }

            const double* net_ptr = static_cast<const double*>(net_info.ptr);
            double min_x = std::numeric_limits<double>::infinity();
            double min_y = std::numeric_limits<double>::infinity();
            double max_x = -std::numeric_limits<double>::infinity();
            double max_y = -std::numeric_limits<double>::infinity();

            for (ssize_t i = 0; i < pin_num; ++i) {
                const int cell_idx = static_cast<int>(std::llround(net_ptr[i * 3]));
                if (cell_idx < 0 || cell_idx >= cell_num) {
                    throw std::runtime_error("net contains a cell index out of range.");
                }
                const double pin_x = sol_ptr[cell_idx * 2] + net_ptr[i * 3 + 1];
                const double pin_y = sol_ptr[cell_idx * 2 + 1] + net_ptr[i * 3 + 2];
                min_x = std::min(min_x, pin_x);
                min_y = std::min(min_y, pin_y);
                max_x = std::max(max_x, pin_x);
                max_y = std::max(max_y, pin_y);
            }

            const double bbox_w = max_x - min_x;
            const double bbox_h = max_y - min_y;
            const double net_hpwl = bbox_w + bbox_h;
            hpwl += net_hpwl;
            if (net_hpwl <= 0.0) {
                continue;
            }

            const double rudy_w = std::max(bbox_w, bin_w);
            const double rudy_h = std::max(bbox_h, bin_h);
            const double bbox_area = rudy_w * rudy_h;
            const double density = (rudy_w + rudy_h) / bbox_area;

            const double center_x = 0.5 * (min_x + max_x);
            const double center_y = 0.5 * (min_y + max_y);
            const double rudy_xl = std::max(die_xl, center_x - 0.5 * rudy_w);
            const double rudy_xh = std::min(die_xh, center_x + 0.5 * rudy_w);
            const double rudy_yl = std::max(die_yl, center_y - 0.5 * rudy_h);
            const double rudy_yh = std::min(die_yh, center_y + 0.5 * rudy_h);
            if (rudy_xh <= rudy_xl || rudy_yh <= rudy_yl) {
                continue;
            }

            const int c0 = clamp_bin(rudy_xl, die_xl, bin_w, cols);
            const int c1 = clamp_bin(std::nextafter(rudy_xh, -std::numeric_limits<double>::infinity()), die_xl, bin_w, cols);
            const int r0 = clamp_bin(rudy_yl, die_yl, bin_h, rows);
            const int r1 = clamp_bin(std::nextafter(rudy_yh, -std::numeric_limits<double>::infinity()), die_yl, bin_h, rows);
            for (int row = r0; row <= r1; ++row) {
                const double by0 = die_yl + row * bin_h;
                const double by1 = by0 + bin_h;
                const double overlap_h = std::max(0.0, std::min(rudy_yh, by1) - std::max(rudy_yl, by0));
                if (overlap_h <= 0.0) {
                    continue;
                }
                for (int col = c0; col <= c1; ++col) {
                    const double bx0 = die_xl + col * bin_w;
                    const double bx1 = bx0 + bin_w;
                    const double overlap_w = std::max(0.0, std::min(rudy_xh, bx1) - std::max(rudy_xl, bx0));
                    if (overlap_w <= 0.0) {
                        continue;
                    }
                    rudy_map[static_cast<size_t>(row) * cols + col] += density * (overlap_w * overlap_h) / bin_area;
                }
            }
        }

        py::array_t<double> congestion_map({rows, cols});
        std::copy(
            rudy_map.begin(),
            rudy_map.end(),
            static_cast<double*>(congestion_map.mutable_data())
        );
        return py::make_tuple(static_cast<double>(hpwl), congestion_map);
    }
};

PYBIND11_MODULE(edap_helper_impl, m) {
    m.doc() = "PyBind11 helper for EDA placement constraints and evaluation";
    py::class_<EDAHelper>(m, "EDAHelper")
        .def(py::init<>())
        .def(
            "check_constraints",
            &EDAHelper::check_constraints,
            py::arg("sol"),
            py::arg("die"),
            py::arg("cells"),
            py::arg("macro_mask") = py::none()
        )
        .def(
            "evaluate",
            &EDAHelper::evaluate,
            py::arg("sol"),
            py::arg("nets"),
            py::arg("die"),
            py::arg("bin_cols") = 224,
            py::arg("bin_rows") = 224
        );
}
