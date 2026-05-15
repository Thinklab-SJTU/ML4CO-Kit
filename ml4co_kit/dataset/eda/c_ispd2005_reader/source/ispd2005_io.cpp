#include <cctype>
#include <fstream>
#include <limits>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct ParsedPin {
    ssize_t cell_idx;
    std::string cell_name;
    double offset_x;
    double offset_y;
};

std::string trim(const std::string& value) {
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
        ++first;
    }

    size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) {
        --last;
    }

    return value.substr(first, last - first);
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool is_skippable_line(const std::string& line) {
    if (line.empty() || line[0] == '#') {
        return true;
    }
    return starts_with(line, "UCLA") ||
           starts_with(line, "NumNodes") ||
           starts_with(line, "NumTerminals") ||
           starts_with(line, "NumNets") ||
           starts_with(line, "NumPins");
}

bool looks_like_o_index(const std::string& name) {
    if (name.size() < 2 || name[0] != 'o') {
        return false;
    }
    for (size_t i = 1; i < name.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
            return false;
        }
    }
    return true;
}

}  // namespace

class ISPD2005Reader {
public:
    py::tuple from_nodes(const std::string& file_path) {
        std::ifstream fin(file_path);
        if (!fin.is_open()) {
            throw std::runtime_error("failed to open nodes file: " + file_path);
        }

        name_to_idx_.clear();
        node_sizes_.clear();
        std::vector<double> sizes;
        std::vector<bool> terminal_mask;

        std::string raw_line;
        while (std::getline(fin, raw_line)) {
            const std::string line = trim(raw_line);
            if (is_skippable_line(line)) {
                continue;
            }

            std::istringstream iss(line);
            std::string name;
            double width = 0.0;
            double height = 0.0;
            std::string node_type;
            if (!(iss >> name >> width >> height)) {
                continue;
            }
            iss >> node_type;

            const ssize_t idx = static_cast<ssize_t>(terminal_mask.size());
            name_to_idx_[name] = idx;
            sizes.push_back(width);
            sizes.push_back(height);
            node_sizes_.push_back(width);
            node_sizes_.push_back(height);
            terminal_mask.push_back(starts_with(node_type, "terminal"));
        }

        const ssize_t node_num = static_cast<ssize_t>(terminal_mask.size());
        // std::cerr << "from_nodes parsed " << node_num << " nodes" << std::endl;
        py::array_t<double> size_array(std::vector<ssize_t>{node_num, static_cast<ssize_t>(2)});
        // std::cerr << "from_nodes allocated sizes" << std::endl;
        std::copy(sizes.begin(), sizes.end(), static_cast<double*>(size_array.mutable_data()));
        // std::cerr << "from_nodes copied sizes" << std::endl;
        py::list terminal_list;
        for (ssize_t i = 0; i < node_num; ++i) {
            terminal_list.append(py::bool_(terminal_mask[static_cast<size_t>(i)]));
        }
        // std::cerr << "from_nodes built terminal list" << std::endl;
        py::object np = py::module_::import("numpy");
        // std::cerr << "from_nodes imported numpy" << std::endl;
        py::object terminal_array = np.attr("array")(
            terminal_list,
            py::arg("dtype") = np.attr("bool_")
        );
        // std::cerr << "from_nodes built terminal array" << std::endl;

        return py::make_tuple(size_array, terminal_array);
    }

    py::list from_nets(const std::string& file_path) const {
        std::ifstream fin(file_path);
        if (!fin.is_open()) {
            throw std::runtime_error("failed to open nets file: " + file_path);
        }

        py::list nets;
        std::string raw_line;
        while (std::getline(fin, raw_line)) {
            const std::string line = trim(raw_line);
            if (is_skippable_line(line)) {
                continue;
            }

            if (!starts_with(line, "NetDegree")) {
                continue;
            }

            std::string label;
            std::string colon;
            ssize_t degree = 0;
            std::string net_name;
            std::istringstream net_iss(line);
            if (!(net_iss >> label >> colon >> degree)) {
                throw std::runtime_error("failed to parse NetDegree line: " + line);
            }
            net_iss >> net_name;

            std::vector<ParsedPin> pins;
            pins.reserve(static_cast<size_t>(degree));

            for (ssize_t pin_idx = 0; pin_idx < degree; ++pin_idx) {
                if (!std::getline(fin, raw_line)) {
                    throw std::runtime_error("unexpected EOF while reading pins of net: " + net_name);
                }

                const std::string pin_line = trim(raw_line);
                std::istringstream pin_iss(pin_line);
                std::string cell_name;
                std::string direction;
                std::string pin_colon;
                double offset_x = 0.0;
                double offset_y = 0.0;
                if (!(pin_iss >> cell_name >> direction >> pin_colon >> offset_x >> offset_y)) {
                    throw std::runtime_error("failed to parse pin line: " + pin_line);
                }

                const ssize_t idx = cell_index(cell_name);
                const size_t size_offset = static_cast<size_t>(idx) * 2;
                if (size_offset + 1 >= node_sizes_.size()) {
                    throw std::runtime_error(
                        "cell size for '" + cell_name + "' is unavailable. "
                        "Call from_nodes before from_nets."
                    );
                }

                // Bookshelf pin offsets are relative to the node center; the
                // evaluator uses offsets from the node lower-left corner.
                pins.push_back(ParsedPin{
                    idx,
                    cell_name,
                    offset_x + 0.5 * node_sizes_[size_offset],
                    offset_y + 0.5 * node_sizes_[size_offset + 1]
                });
            }

            std::sort(pins.begin(), pins.end(), [](const ParsedPin& a, const ParsedPin& b) {
                return a.cell_name < b.cell_name;
            });
            const auto last = std::unique(
                pins.begin(),
                pins.end(),
                [](const ParsedPin& a, const ParsedPin& b) {
                    return a.cell_name == b.cell_name;
                }
            );
            pins.erase(last, pins.end());

            py::array_t<double> net_array(std::vector<ssize_t>{
                static_cast<ssize_t>(pins.size()),
                static_cast<ssize_t>(3)
            });
            double* net_ptr = static_cast<double*>(net_array.mutable_data());
            for (size_t pin_idx = 0; pin_idx < pins.size(); ++pin_idx) {
                net_ptr[pin_idx * 3] = static_cast<double>(pins[pin_idx].cell_idx);
                net_ptr[pin_idx * 3 + 1] = pins[pin_idx].offset_x;
                net_ptr[pin_idx * 3 + 2] = pins[pin_idx].offset_y;
            }
            nets.append(net_array);
        }

        return nets;
    }

    py::array_t<double> from_lg_pl(const std::string& file_path) {
        std::ifstream fin(file_path);
        if (!fin.is_open()) {
            throw std::runtime_error("failed to open pl file: " + file_path);
        }

        const bool has_node_index = !name_to_idx_.empty();
        std::vector<double> positions;
        if (has_node_index) {
            positions.assign(name_to_idx_.size() * 2, std::numeric_limits<double>::quiet_NaN());
        }

        std::string raw_line;
        ssize_t next_idx = 0;
        while (std::getline(fin, raw_line)) {
            const std::string line = trim(raw_line);
            if (line.empty() || line[0] == '#' || starts_with(line, "UCLA")) {
                continue;
            }

            std::istringstream iss(line);
            std::string name;
            double x = 0.0;
            double y = 0.0;
            if (!(iss >> name >> x >> y)) {
                continue;
            }

            ssize_t idx = next_idx;
            if (has_node_index) {
                idx = cell_index(name);
            } else {
                name_to_idx_[name] = idx;
                positions.push_back(x);
                positions.push_back(y);
                ++next_idx;
                continue;
            }

            positions[static_cast<size_t>(idx) * 2] = x;
            positions[static_cast<size_t>(idx) * 2 + 1] = y;
        }

        const ssize_t node_num = static_cast<ssize_t>(positions.size() / 2);
        py::array_t<double> position_array(std::vector<ssize_t>{node_num, static_cast<ssize_t>(2)});
        std::copy(positions.begin(), positions.end(), static_cast<double*>(position_array.mutable_data()));
        return position_array;
    }

private:
    ssize_t cell_index(const std::string& name) const {
        const auto it = name_to_idx_.find(name);
        if (it != name_to_idx_.end()) {
            return it->second;
        }
        if (looks_like_o_index(name)) {
            return static_cast<ssize_t>(std::stoll(name.substr(1)));
        }
        throw std::runtime_error(
            "unknown cell name '" + name + "'. Call from_nodes before from_nets/from_lg_pl "
            "for non-oN Bookshelf names."
        );
    }

    std::unordered_map<std::string, ssize_t> name_to_idx_;
    std::vector<double> node_sizes_;
};

PYBIND11_MODULE(ispd2005_io_impl, m) {
    m.doc() = "PyBind11 IO reader for ISPD2005 Bookshelf placement files";
    py::class_<ISPD2005Reader>(m, "ISPD2005Reader")
        .def(py::init<>())
        .def("from_nodes", &ISPD2005Reader::from_nodes, py::arg("file_path"))
        .def("from_nets", &ISPD2005Reader::from_nets, py::arg("file_path"))
        .def("from_lg_pl", &ISPD2005Reader::from_lg_pl, py::arg("file_path"));
}
