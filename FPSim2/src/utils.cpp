#include "utils.hpp"
#include "popcnt.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;

namespace utils {

    uint64_t PyPopcount(const py::array_t<uint64_t> py_query) {
        const auto query = py_query.unchecked<1>();
        uint64_t qcount = 0;
        for (ssize_t i = 0; i < query.shape(0); i++)
            qcount += popcntll(query(i));
        return qcount;
    }

    py::list BitStrToIntList(const std::string &bit_string) {
        py::list efp;
        size_t len = bit_string.length();
        for (size_t i = 0; i < len; i += 64) {
            uint64_t value = 0;
            for (size_t j = 0; j < 64 && (i + j) < len; ++j) {
                value = (value << 1) | (bit_string[i + j] - '0');
            }
            efp.append(value);
        }
        return efp;
    }

    bool cmp(const Result &l, const Result &r) {
        return l.coeff > r.coeff;
    }

    void SortResults(py::array_t<Result> py_res) {
        auto res = py_res.unchecked<1>();
        Result *ptr = (Result *)res.data(0);
        std::sort(&ptr[0], &ptr[res.shape(0)], cmp);
    }
}
