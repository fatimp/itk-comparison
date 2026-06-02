cmake_minimum_required(VERSION 3.16.3)

project(itk_example)

link_directories(/usr/local/lib)
find_package(ITK REQUIRED)
find_package(GTest REQUIRED)
include( ${ITK_USE_FILE} )
add_executable(itk itk.cpp)
target_include_directories(itk PUBLIC ${ITK_INCLUDE_DIRS})
target_link_libraries(itk ${ITK_LIBRARIES} GTest::GTest)
