# CMake module to configure wgpu-native (WebGPU C bindings) for openEMS
# Supports macOS (Metal), Linux (Vulkan), and Windows (DirectX 12 / Vulkan)

if(NOT ENABLE_WEBGPU)
    return()
endif()

message(STATUS "Configuring WebGPU acceleration backend (wgpu-native)...")

# Detect platform architecture
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(WGPU_ARCH "macos-aarch64")
    else()
        set(WGPU_ARCH "macos-x86_64")
    endif()
elseif(WIN32)
    set(WGPU_ARCH "windows-x86_64-msvc")
elseif(UNIX)
    set(WGPU_ARCH "linux-x86_64")
endif()

# Check for local or system wgpu-native installation
set(WEBGPU_SEARCH_PATHS
    ${WEBGPU_ROOT_DIR}
    ${CMAKE_BINARY_DIR}/wgpu_test
    ${PROJECT_SOURCE_DIR}/build/wgpu_test
    ${PROJECT_SOURCE_DIR}/wgpu_native
)

find_path(WEBGPU_INCLUDE_DIR
    NAMES webgpu/webgpu.h webgpu.h
    PATHS
        ${WEBGPU_SEARCH_PATHS}
    PATH_SUFFIXES
        include
        /usr/local/include
        /opt/homebrew/include
)

find_library(WEBGPU_LIBRARY
    NAMES wgpu_native wgpu
    PATHS
        ${WEBGPU_SEARCH_PATHS}
    PATH_SUFFIXES
        lib
        bin
        /usr/local/lib
        /opt/homebrew/lib
)

if(WEBGPU_INCLUDE_DIR AND WEBGPU_LIBRARY)
    message(STATUS "Found existing wgpu-native library: ${WEBGPU_LIBRARY}")
    message(STATUS "Found existing wgpu-native include: ${WEBGPU_INCLUDE_DIR}")
    include_directories(${WEBGPU_INCLUDE_DIR})
    set(WEBGPU_LIBRARIES ${WEBGPU_LIBRARY})
    add_definitions(-DENABLE_WEBGPU)

    if(WIN32)
        find_file(WEBGPU_DLL
            NAMES wgpu_native.dll
            PATHS
                ${WEBGPU_SEARCH_PATHS}
            PATH_SUFFIXES
                lib
                bin
        )
        if(WEBGPU_DLL)
            message(STATUS "Found wgpu-native DLL: ${WEBGPU_DLL}")
            file(COPY ${WEBGPU_DLL} DESTINATION ${CMAKE_BINARY_DIR})
            install(FILES ${WEBGPU_DLL} DESTINATION bin)
        endif()
    endif()
else()
    message(STATUS "wgpu-native target platform: ${WGPU_ARCH}")
    message(STATUS "To link against wgpu-native, set -DWEBGPU_ROOT_DIR=<path_to_wgpu_native>")
    # Enable fallback stub mode when prebuilt binary is not present during configuration
    add_definitions(-DENABLE_WEBGPU_STUB)
endif()
