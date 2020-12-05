cmake_minimum_required(VERSION 2.8)
project( GsColorbook )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( colorbook colorbook.cpp )
target_link_libraries( colorbook ${OpenCV_LIBS} )