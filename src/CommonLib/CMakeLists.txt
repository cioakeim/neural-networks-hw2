#find_package(OpenMP REQUIRED)
#include_directories(${OpenMP_INCLUDE_DIRS})

#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/third-party/eigen-3.4.0)

find_package(MKL REQUIRED)

file(GLOB COMMON_LIB_SRC "${PROJECT_SOURCE_DIR}/src/CommonLib/*.cpp")
add_library(common_lib STATIC ${COMMON_LIB_SRC})
target_compile_options(common_lib PUBLIC -O3 -ffast-math -Wall -Wextra)
target_include_directories(common_lib PUBLIC "${PROJECT_SOURCE_DIR}/include/CommonLib")
#target_link_libraries(common_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(common_lib PUBLIC MKL::MKL)
#target_link_libraries(common_lib PUBLIC ${OpenCV_LIBS})
#target_link_libraries(common_lib PUBLIC ${AOCL_BLAS_LIB})
#target_link_libraries(common_lib PUBLIC ${AOCL_LAPACK_LIB})



