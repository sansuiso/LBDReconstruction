# -----------------------------------------------
# NON-BINARIZED LBD RECONSTRUCTION
# -----------------------------------------------

set (BIHT_SRC_FILES main_lbdrec_biht.cpp)
add_executable (lbdrec_biht ${BIHT_SRC_FILES})
add_dependencies (lbdrec_biht LBDReconstruction)
target_link_libraries (lbdrec_biht LBDReconstruction ${LBD_EXT_LIBRARIES})

# -----------------------------------------------
# INSTALL BINARIES
# -----------------------------------------------

install (TARGETS lbdrec_biht RUNTIME DESTINATION bin)
