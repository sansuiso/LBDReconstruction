# -----------------------------------------------
# BUILD THE LBD LIBRARY
# -----------------------------------------------
include_directories (${CMAKE_CURRENT_SOURCE_DIR})

set (LBDREC_SRC_FILES BriefOperator.cpp
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

# Add library target
add_library (LBDReconstruction ${LBDREC_SRC_FILES})
target_link_libraries (LBDReconstruction ${LBD_EXT_LIBRARIES} )
