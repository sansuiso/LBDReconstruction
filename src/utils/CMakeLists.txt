# -----------------------------------------------
# VARIOUS TESTS AND MORE-OR-LESS USEFUL STUFF
# -----------------------------------------------

# Compute operator norms

add_executable (lbd_operator_norms main_lbd_operator_norms.cpp)
target_link_libraries (lbd_operator_norms PRIVATE LBDReconstruction)

# Draw as circles

add_executable (lbd_draw_circles main_draw_lbd_circles.cpp)
target_link_libraries (lbd_draw_circles PRIVATE LBDReconstruction)

# Merge 2 images

add_executable (lbd_merge2image main_lbd_merge2image.cpp)
target_link_libraries (lbd_merge2image PRIVATE LBDReconstruction)

# Merge 2 images into a movie

add_executable (lbd_merge2movie main_lbd_merge2movie.cpp)
target_link_libraries (lbd_merge2movie PRIVATE LBDReconstruction)

# Cumulative image

add_executable (lbd_cumul_image main_lbd_create_cumulative_image.cpp)
target_link_libraries (lbd_cumul_image PRIVATE LBDReconstruction)
