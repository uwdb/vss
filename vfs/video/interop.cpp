#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mp4.h"

PYBIND11_PLUGIN(interop)
{
    pybind11::module module("interop");
    module.def("unmux", &unmux);
    module.def("write", &write_raw);
    module.def("open", &open_output);

    return module.ptr();
}