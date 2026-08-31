cmake_minimum_required(VERSION 3.26)
project(avap_core LANGUAGES CXX)

option(AVAP_WITH_HIP "Build the HIP bridge and kernels (requires ROCm)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBVA REQUIRED IMPORTED_TARGET libva libva-drm)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavformat libavutil)
pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm)

set(AVAP_SOURCES
    cpp/module.cpp
    cpp/vaapi_decoder.cpp
)

if(AVAP_WITH_HIP)
    # ROCm ships CMake configs under /opt/rocm
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    enable_language(HIP)
    find_package(hip REQUIRED)
    list(APPEND AVAP_SOURCES cpp/hip_bridge.cpp cpp/kernels.hip)
    set_source_files_properties(cpp/kernels.hip PROPERTIES LANGUAGE HIP)
endif()

# NO_EXTRAS: pybind11's default LTO produces GCC slim-LTO objects that the
# ROCm clang link step (used because kernels.hip is in the target) silently
# discards, leaving a module without PyInit__core.
pybind11_add_module(_core NO_EXTRAS ${AVAP_SOURCES})

target_link_libraries(_core PRIVATE
    PkgConfig::LIBVA
    PkgConfig::FFMPEG
    PkgConfig::LIBDRM
)

if(AVAP_WITH_HIP)
    target_compile_definitions(_core PRIVATE AVAP_WITH_HIP=1)
    target_link_libraries(_core PRIVATE hip::host)
endif()

install(TARGETS _core LIBRARY DESTINATION avap)
