#find_package(OpenMP REQUIRED)
#include_directories(${OpenMP_INCLUDE_DIRS})

option(OSQP_MANUAL_LINK "Set to On if OSQP isn't in default location" OFF)


find_package(MKL REQUIRED)
find_package(osqp REQUIRED)


include_directories(${MY_HOME_DIR}/my_libroot/include)
include_directories(${PROJECT_SOURCE_DIR}/include/CommonLib)

file(GLOB SVM_SOURCES "${PROJECT_SOURCE_DIR}/src/SVM/*.cpp")
add_library(svm_lib STATIC ${SVM_SOURCES})
target_compile_options(svm_lib PUBLIC -O3 -ffast-math -Wall -Wextra)
#target_link_libraries(svm_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(svm_lib PUBLIC common_lib)
target_link_libraries(svm_lib PUBLIC MKL::MKL)
#target_link_libraries(svm_lib PUBLIC ${AOCL_BLAS_LIB})
#target_link_libraries(svm_lib PUBLIC ${AOCL_LAPACK_LIB})
target_link_libraries(svm_lib PUBLIC osqp::osqp)
