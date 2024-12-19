find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})

option(OSQP_MANUAL_LINK "Set to On if OSQP isn't in default location" OFF)

set(OSQP_LIBROOT /home/c/cioakeim/osqp/)

if(OSQP_MANUAL_LINK)
  include_directories(${OSQP_LIBROOT}/include) 
  link_directories(${OSQP_LIBROOT}/lib)
else()
  find_package(osqp REQUIRED)
endif()

#find_package(BLAS REQUIRED)
set(BLAS_LIBRARIES /home/chris/aocl/5.0.0/aocc/lib/libblis.so)

include_directories(${PROJECT_SOURCE_DIR}/include/CommonLib)

file(GLOB SVM_SOURCES "${PROJECT_SOURCE_DIR}/src/SVM/*.cpp")
add_library(svm_lib STATIC ${SVM_SOURCES})
target_compile_options(svm_lib PUBLIC -O3 -ffast-math -Wall -Wextra)
target_link_libraries(svm_lib PUBLIC ${BLAS_LIBRARIES})
target_link_libraries(svm_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(svm_lib PUBLIC common_lib)
if(OSQP_MANUAL_LINK)
  target_link_libraries(svm_lib PUBLIC ${OSQP_LIBROOT}/lib/libosqp.so)
else()
  target_link_libraries(svm_lib PUBLIC osqp::osqp)
endif()
