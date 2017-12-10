#include "ratings_util.h"

SimilarityMatrix computeSimilarityGold(RatingsMatrixCSR &ratingMatrix);

float calculatePearsonCorrelationGold(RatingsMatrixCSR &ratingsMatrix, unsigned int user1, unsigned int user2);

vector<ItemRating> calculateTopNRecommendationsForUserGold(RatingsMatrixCSR &ratingsMatrix,
                                                           SimilarityMatrix &similarityMatrix,
                                                           vector<unsigned int> &movieIds,
                                                           unsigned int userId,
                                                           unsigned int n);

int binaryLocate(vector<unsigned int> &array, int l, int r, unsigned int target);