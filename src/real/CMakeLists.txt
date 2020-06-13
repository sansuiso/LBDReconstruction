# -----------------------------------------------
# NON-BINARIZED LBD RECONSTRUCTION
# -----------------------------------------------

add_executable (lbdrec_wl1 main_lbdrec_primaldual_wl1.cpp)
target_link_libraries (lbdrec_wl1 PRIVATE LBDReconstruction)

add_executable (lbdrec_tvl1 main_lbdrec_primaldual_tvl1.cpp)
target_link_libraries (lbdrec_tvl1 PRIVATE LBDReconstruction)

# -----------------------------------------------
# INSTALL BINARIES
# -----------------------------------------------

install (TARGETS lbdrec_tvl1 lbdrec_wl1 RUNTIME DESTINATION bin)
