find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include/CommonLib)

file(GLOB MLP_SOURCES "${PROJECT_SOURCE_DIR}/src/MLP/*.cpp")
add_library(mlp_lib STATIC ${MLP_SOURCES})
target_link_libraries(mlp_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(mlp_lib PUBLIC common_lib)


