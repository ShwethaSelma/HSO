set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/dpct_migrated/Common/)

add_executable (dpct_migrated flowCUDA.cpp flowGold.cpp main.cpp)
target_link_libraries(dpct_migrated sycl)

add_custom_target (run_sdm_cpu cd ${CMAKE_SOURCE_DIR} && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dpct_migrated) 
add_custom_target (run_sdm_gpu SYCL_DEVICE_FILTER=gpu cd ${CMAKE_SOURCE_DIR} && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dpct_migrated)