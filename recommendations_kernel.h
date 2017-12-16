#include "ratings_util.h"

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define BLOCK_SIZE 64
#define TILE_SIZE 512 // assumption max TILE_SIZE ratings per row

SimilarityMatrix computeSimilarityParallel(RatingsMatrixCSR &ratingMatrix);
