cmake_minimum_required(VERSION 3.20)

#----------------------------------------------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CUDA_HOST_COMPILER "g++")

#----------------------------------------------------------------------------------------------------------------------

find_package(OpenMP)

#----------------------------------------------------------------------------------------------------------------------

find_package(Eigen3 REQUIRED)

#----------------------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wall -O2 -fPIC -ffast-math -mavx")

#----------------------------------------------------------------------------------------------------------------------

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB main_source ${PROJECT_SOURCE_DIR}/main.cc)

#----------------------------------------------------------------------------------------------------------------------

add_executable(main ${main_source} ${sources})
target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

#----------------------------------------------------------------------------------------------------------------------

set(MAIN_SCRIPTS )

foreach(_script ${MAIN_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
    )
endforeach()

#----------------------------------------------------------------------------------------------------------------------

install(TARGETS main DESTINATION ${PROJECT_SOURCE_DIR}/install)

#----------------------------------------------------------------------------------------------------------------------

add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)

#----------------------------------------------------------------------------------------------------------------------