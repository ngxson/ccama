add_subdirectory(llama.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_SRC experiments/save_state_quant.cpp
    actions.hpp
    json.hpp
    llama.cpp/llama.cpp
    llama.cpp/llama.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common)

add_executable(save_state_quant experiments/save_state_quant.cpp ${COMMON_SRC})
target_link_libraries(save_state_quant PRIVATE ggml common ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(experiments)
add_dependencies(experiments save_state_quant)
