#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <vector>

using namespace std;

// for sorting composite item-wise ratings

typedef struct ItemRating {
    unsigned int item;
    float rating;

    bool operator<(ItemRating &itemRating) {
        return (item < itemRating.item);
    }
} ItemRating;

typedef struct {
    vector<float> data;
    vector<unsigned int> cols;
    vector<unsigned int> rowPtrs;
    vector<unsigned int> userIds;
    vector<float> userEuclideanNorm;
    vector<float> userMean;
} RatingsMatrixCSR;

RatingsMatrixCSR *readInputRatings(string file);

void displayRatingMatrix(RatingsMatrixCSR &ratingMatrix);

void normalizeRatingVectors(RatingsMatrixCSR &ratingsMatrix);

typedef struct SimilarityMatrix {
    float *similarities;
    unsigned int length;
    unsigned int width;
} SimilarityMatrix;

void initSimilarityMatrix(SimilarityMatrix &similarityMatrix);

void displaySimilarityMatrix(SimilarityMatrix &similarityMatrix);