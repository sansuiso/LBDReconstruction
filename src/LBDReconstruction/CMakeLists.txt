# -----------------------------------------------
# BUILD THE LBD LIBRARY
# -----------------------------------------------
include_directories (${CMAKE_CURRENT_SOURCE_DIR})

# Add library target
add_library (LBDReconstruction "")
target_sources(LBDReconstruction
    PRIVATE
        BriefOperator.cpp
        ExFreakOperator.cpp
        FreakOperator.cpp
        ILBDOperator.cpp
        RandomFreakOperator.cpp
        SignOperator.cpp
        reconstruction.cpp
        wavelets.cpp
        prox.cpp
        derivatives.cpp
)
target_link_libraries (LBDReconstruction
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        ${LBD_EXT_LIBRARIES}
)
