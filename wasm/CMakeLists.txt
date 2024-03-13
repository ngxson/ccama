project("ccama")
add_subdirectory(../llama.cpp llamacpp)

set(COMMON_SRC main.cpp
    ../actions.hpp
    ../json.hpp
    ../llama.cpp/llama.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/common)

add_executable(wllama main.cpp ${COMMON_SRC})
target_link_libraries(wllama PRIVATE ggml common ${CMAKE_THREAD_LIBS_INIT})
