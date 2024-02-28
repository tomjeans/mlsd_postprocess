#all files in
cmake_minimum_required(VERSION 3.0)
project(Installation VERSION 1.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -g -Wl,--no-as-needed")
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)



#reduce the space
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")





PROJECT(mlsd_post)

aux_source_directory(./src DIR_SRCS)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)#



include_directories(${PROJECT_SOURCE_DIR}/)
include_directories(${PROJECT_SOURCE_DIR}/include)#basic library
include_directories(${PROJECT_SOURCE_DIR}/eigen-3.4.0)#eigen library

message(STATUS ${PROJECT_SOURCE_DIR}"----the path source dir")




#可执行文件
add_executable(${PROJECT_NAME} ${DIR_SRCS})
target_link_libraries(${PROJECT_NAME} -latomic -pthread -ldl -lz)
#add_library(${PROJECT_NAME} SHARED ${DIR_SRCS})



#cxx flags
target_compile_options(${PROJECT_NAME} PRIVATE -std=c++14 -ffast-math -Ofast -Os -DNDEBUG -fno-exceptions -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables -fexceptions -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections)




message(STATUS ${PROJECT_SOURCE_DIR}"----complete complile and product the out file")