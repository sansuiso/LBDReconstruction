# -----------------------------------------------
# NON-BINARIZED LBD RECONSTRUCTION
# -----------------------------------------------

add_executable (lbdrec_biht main_lbdrec_biht.cpp)
target_link_libraries (lbdrec_biht PRIVATE LBDReconstruction)

# -----------------------------------------------
# INSTALL BINARIES
# -----------------------------------------------

install (TARGETS lbdrec_biht RUNTIME DESTINATION bin)
