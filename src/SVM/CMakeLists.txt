find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include/CommonLib)

file(GLOB MLP_SOURCES "${PROJECT_SOURCE_DIR}/src/SVM/*.cpp")
add_library(svm_lib STATIC ${SVM_SOURCES})
target_link_libraries(svm_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(svm_lib PUBLIC common_lib)

