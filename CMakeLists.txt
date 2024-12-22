cmake_minimum_required(VERSION 3.10)

project(NeuralNetworksProject2 VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


include_directories(${PROJECT_SOURCE_DIR}/include)

set(MY_HOME_DIR "/home/chris" CACHE PATH "Home Dir")
set(AOCL_BLAS_LIB "${MY_HOME_DIR}/aocl/5.0.0/aocc/lib/libblis-mt.so" CACHE PATH "BLAS")
set(AOCL_LAPACK_LIB "${MY_HOME_DIR}/aocl/5.0.0/aocc/lib/libflame.so" CACHE PATH "LAPACK")
set(ENABLE_MULTITHREADING ON CACHE STRING "MT")

add_subdirectory("${PROJECT_SOURCE_DIR}/src/CommonLib")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/MLP")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/SVM")

# Standard CMake Structure 

include_directories(${MY_HOME_DIR}/my_libroot/include)

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

add_executable(test2ClassSVM "${SCRIPTS}/test2ClassSVM.cpp")
target_link_libraries(test2ClassSVM PRIVATE common_lib)
target_link_libraries(test2ClassSVM PRIVATE svm_lib)
target_compile_options(test2ClassSVM PRIVATE -O3 -Wall -Wextra)

add_executable(testMultiSVM "${SCRIPTS}/testMultiSVM.cpp")
target_link_libraries(testMultiSVM PRIVATE common_lib)
target_link_libraries(testMultiSVM PRIVATE svm_lib)
target_compile_options(testStoredModel PRIVATE -O3 -Wall -Wextra)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)


