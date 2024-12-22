find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})

option(OSQP_MANUAL_LINK "Set to On if OSQP isn't in default location" OFF)

set(MY_OSQP_LIBROOT /home/c/cioakeim/osqp/)

find_package(MKL REQUIRED)


if(OSQP_MANUAL_LINK)
  include_directories(${MY_OSQP_LIBROOT}/include) 
  link_directories(${MY_OSQP_LIBROOT}/lib64)
else()
  find_package(osqp REQUIRED)
endif()


include_directories(${PROJECT_SOURCE_DIR}/include/CommonLib)

file(GLOB SVM_SOURCES "${PROJECT_SOURCE_DIR}/src/SVM/*.cpp")
add_library(svm_lib STATIC ${SVM_SOURCES})
target_compile_options(svm_lib PUBLIC -O3 -ffast-math -Wall -Wextra)
target_link_libraries(svm_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(svm_lib PUBLIC common_lib)
target_link_libraries(svm_lib PUBLIC MKL::MKL)
target_link_libraries(svm_lib PUBLIC ${AOCL_BLAS_LIB})
target_link_libraries(svm_lib PUBLIC ${AOCL_LAPACK_LIB})
if(OSQP_MANUAL_LINK)
  target_include_directories(svm_lib PUBLIC ${OSQP_LIBROOT}/include)
  target_link_libraries(svm_lib PUBLIC ${OSQP_LIBROOT}/lib64/libosqp.so)
else()
  target_link_libraries(svm_lib PUBLIC osqp::osqp)
endif()
