#include "ratings_util.h"

SimilarityMatrix computeSimilarity(RatingsMatrixCSR &ratingMatrix);

float calculatePearsonCorrelation(RatingsMatrixCSR &ratingsMatrix, unsigned int user1, unsigned int user2);