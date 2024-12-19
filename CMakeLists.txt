cmake_minimum_required(VERSION 3.10)

project(NeuralNetworksProject1 VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third-party/eigen-3.4.0)

set(BLAS_LIBRARIES "/home/chris/aocl/5.0.0/aocc/lib/libblis.so" CACHE PATH "BLAS")

add_subdirectory("${PROJECT_SOURCE_DIR}/src/CommonLib")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/MLP")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/SVM")

# Standard CMake Structure 


# Scripts folder
set(SCRIPTS "${PROJECT_SOURCE_DIR}/scripts")



add_executable(testMLP "${SCRIPTS}/testMLP.cpp")
target_link_libraries(testMLP PRIVATE mlp_lib)
target_link_libraries(testMLP PRIVATE common_lib)
target_compile_options(testMLP PRIVATE -O3 -Wall -Wextra)

add_executable(testMLP_Adam "${SCRIPTS}/testMLP_Adam.cpp")
target_link_libraries(testMLP_Adam PRIVATE mlp_lib)
target_link_libraries(testMLP_Adam PRIVATE common_lib)
target_compile_options(testMLP_Adam PRIVATE -O3 -Wall -Wextra)

add_executable(testStoredModel "${SCRIPTS}/testStoredModel.cpp")
target_link_libraries(testStoredModel PRIVATE mlp_lib)
target_link_libraries(testStoredModel PRIVATE common_lib)
target_compile_options(testStoredModel PRIVATE -O3 -Wall -Wextra)

add_executable(trashTest "${SCRIPTS}/trash_test.cpp")
target_link_libraries(trashTest PRIVATE common_lib)
target_link_libraries(trashTest PRIVATE svm_lib)
target_compile_options(testStoredModel PRIVATE -O3 -Wall -Wextra)

add_executable(testMultiSVM "${SCRIPTS}/testMultiSVM.cpp")
target_link_libraries(testMultiSVM PRIVATE common_lib)
target_link_libraries(testMultiSVM PRIVATE svm_lib)
target_compile_options(testStoredModel PRIVATE -O3 -Wall -Wextra)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)


