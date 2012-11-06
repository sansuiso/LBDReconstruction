# -----------------------------------------------
# VARIOUS TESTS AND MORE-OR-LESS USEFUL STUFF
# -----------------------------------------------

# Compute operator norms

set (LBD_NORM_SRC_FILES main_lbd_operator_norms.cpp)
add_executable (lbd_operator_norms ${LBD_NORM_SRC_FILES})
add_dependencies (lbd_operator_norms LBDReconstruction)
target_link_libraries (lbd_operator_norms LBDReconstruction ${LBD_EXT_LIBRARIES})

# Draw as circles

set (LBD_CIRCLES_SRC_FILES main_draw_lbd_circles.cpp)
add_executable (lbd_draw_circles ${LBD_NORM_SRC_FILES})
add_dependencies (lbd_draw_circles LBDReconstruction)
target_link_libraries (lbd_draw_circles LBDReconstruction ${LBD_EXT_LIBRARIES})

# Merge 2 images

set (LBD_MERGES_SRC_FILES main_lbd_merge2image.cpp)
add_executable (lbd_merge2image ${LBD_MERGES_SRC_FILES})
add_dependencies (lbd_merge2image LBDReconstruction)
target_link_libraries (lbd_merge2image LBDReconstruction ${LBD_EXT_LIBRARIES})

# Merge 2 images into a movie

set (LBD_MERGEMOV_SRC_FILES main_lbd_merge2movie.cpp)
add_executable (lbd_merge2movie ${LBD_MERGEMOV_SRC_FILES})
add_dependencies (lbd_merge2movie LBDReconstruction)
target_link_libraries (lbd_merge2movie LBDReconstruction ${LBD_EXT_LIBRARIES})

# Cumulative image

set (LBD_CUMSUM_SRC_FILES main_lbd_create_cumulative_image.cpp)
add_executable (lbd_cumul_image ${LBD_CUMSUM_SRC_FILES})
add_dependencies (lbd_cumul_image LBDReconstruction)
target_link_libraries (lbd_cumul_image LBDReconstruction ${LBD_EXT_LIBRARIES})

