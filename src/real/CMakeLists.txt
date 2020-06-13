# -----------------------------------------------
# NON-BINARIZED LBD RECONSTRUCTION
# -----------------------------------------------

set (PD_WAVE_SRC_FILES main_lbdrec_primaldual_wl1.cpp)
add_executable (lbdrec_wl1 ${PD_WAVE_SRC_FILES})
target_link_libraries (lbdrec_wl1 LBDReconstruction ${LBD_EXT_LIBRARIES})

set (PD_TV_SRC_FILES main_lbdrec_primaldual_tvl1.cpp)
add_executable (lbdrec_tvl1 ${PD_WAVE_SRC_FILES})
target_link_libraries (lbdrec_tvl1 LBDReconstruction ${LBD_EXT_LIBRARIES})

# -----------------------------------------------
# INSTALL BINARIES
# -----------------------------------------------

install (TARGETS lbdrec_tvl1 lbdrec_wl1 RUNTIME DESTINATION bin)
