cmake_minimum_required(VERSION 3.10)
project(GAMES201_FEM_Jelly_2D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# 1. 关联你电脑上原有的 OpenCV（注意 OpenCV_DIR 建议指向 build 根目录或 cmake 目录）
set(OpenCV_DIR "D:/games101/opencv/build/x64/vc16/lib") 
find_package(OpenCV REQUIRED)

# 2. 关联你电脑上原有的 Eigen 头文件路径
include_directories("D:/games101/eigen-3.4.0")
include_directories(${OpenCV_INCLUDE_DIRS})

# 3. 生成可执行程序（因为这次的代码是单文件极简版，只需要 main.cpp 即可）
add_executable(fem_jelly_2d fem_jelly_2d.cpp)

# 4. 链接 OpenCV 库
target_link_libraries(fem_jelly_2d PRIVATE ${OpenCV_LIBS})

if(WIN32)
    add_custom_command(TARGET fem_jelly_2d POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "D:/games101/opencv/build/x64/vc16/bin/opencv_world4120$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:fem_jelly_2d>"
        VERBATIM
    )
endif()