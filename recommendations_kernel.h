#include "ratings_util.h"

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define BLOCK_SIZE 64
#define TILE_SIZE 512 // assumption max TILE_SIZE ratings per row

SimilarityMatrix computeSimilarityParallel(unsigned int dim, unsigned int *csrRowPtr_d, unsigned int *csrColIdx_d,
                                               float *csrData_d,  float *userEuclideanNorm_d);

void allocateMemoryToDevicePtrs(unsigned int dim, unsigned int **csrRowPtr_d, unsigned int **csrColIdx_d,
                                   float **csrData_d, float **userEuclideanNorm_d, RatingsMatrixCSR &ratingMatrix);

void copyRatingsMatrixToDevicePtrs(unsigned int dim, unsigned int *csrRowPtr_d, unsigned int *csrColIdx_d,  float *csrData_d,
                                       float *userEuclideanNorm_d, RatingsMatrixCSR &ratingMatrix);


vector<ItemRating> calculateTopNRecommendationsForUserParallel(unsigned int *csrRowPtr_d, unsigned int *csrColIdx_d, float *csrData_d,
                                                 SimilarityMatrix similarityMatrix, vector<unsigned int> movieIds,
                                                 RatingsMatrixCSR &ratingMatrix, unsigned int user_id, unsigned int N);

