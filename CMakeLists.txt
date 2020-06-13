# LBD Inverse Problem (aka LBD reconstruction) CMakeLists.txt
cmake_minimum_required(VERSION 3.13)

project(LBDReconstruction)

# -----------------------------------------------
# USER DEFINED VARIABLES
# -----------------------------------------------
option (WITH_DISPATCH "Build with libdispatch (GCD) support for multithreading" OFF)
option (BUILD_DOC     "Build the documentation with Doxygen"                    OFF)

# -----------------------------------------------
# COMPILER FLAGS
# -----------------------------------------------

# Set the build type
if (NOT CMAKE_BUILD_TYPE)
   set (CMAKE_BUILD_TYPE Debug CACHE STRING
        "Choose the type of build. Optionas are: None, Debug, Release, RelWithDebInfo, MinSizeRel"
        FORCE)
endif ()

# Optimize speed in case of Release build
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
endif ()

# Set debug flags in Debug build
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
   add_definitions (-DDEBUG)
endif ()

if (WITH_DISPATCH)
   add_definitions (-DWITH_DISPATCH)
endif()

# -----------------------------------------------
# DEPENDANCIES
# -----------------------------------------------

# Find the dependencies
find_package(OpenCV REQUIRED)

# Include the dependancies
include_directories(SYSTEM ${OPENCV_INCLUDE_DIRS})

set( BUILD_SHARED_LIBS false )

if (WITH_DISPATCH)
   if (APPLE)
      find_library(OSX_SYSTEM_LIB "System")
      set( EXTRA_LIBS ${EXTRA_LIBS} ${OSX_SYSTEM_LIB} )
   else ()
      find_library (DISPATCH_LIB "dispatch")  
   endif (APPLE)
endif(WITH_DISPATCH)

set (LBD_EXT_LIBRARIES ${OpenCV_LIBS} ${EXTRA_LIBS})

# -----------------------------------------------
# SUBDIRECTORIES
# -----------------------------------------------

# -----------------------------------------------
# Library
set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# Add the LBD library
include_directories(src/LBDReconstruction)
add_subdirectory (src/LBDReconstruction)

# -----------------------------------------------
# Add documentation
if (BUILD_DOC)
   find_package(Doxygen)
   configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in ${PROJECT_SOURCE_DIR}/doc/Doxyfile @ONLY)
   add_custom_target(${PROJECT_NAME}_doc ALL ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/Doxyfile)
endif ()

# -----------------------------------------------
# Commands
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_subdirectory (src/utils)
add_subdirectory (src/real)
add_subdirectory (src/binary)
