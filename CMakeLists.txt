cmake_minimum_required(VERSION 3.9)

find_package(Enzyme REQUIRED CONFIG)

# Trade off flexibility for correctness: compilation *will* fail if using a non
# clang compiler, or a different clang/LLVM version than the one used to build
# Enzyme. Compilers can only be configured before a project() call.
message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})
set(CMAKE_C_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang++")

project(libshellexample)

get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
message("imported targets ${importTargetsAfter}")

set(CMAKE_CXX_STANDARD 11)
set(LIBIGL_EIGEN_VERSION 3.3.7 CACHE STRING "Eigen version")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(libigl)
igl_include(glfw)
igl_include(imgui)

find_package(Libshell REQUIRED)

add_executable(libshellexample
    main.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC ${LIBSHELL_INCLUDE_DIR})
target_link_libraries(libshellexample PUBLIC ${LIBSHELL_LIB_DIR} igl::core igl::glfw igl::imgui LLDEnzymeFlags)

