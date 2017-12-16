#ifndef RATINGS_UTIL_H
#define RATINGS_UTIL_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <cstring>
#include <stdexcept>

using namespace std;

// for sorting composite item-wise ratings and recommendations

typedef struct ItemRating {
    unsigned int item;
    float rating;

    bool operator<(const ItemRating &itemRating) const {
        return item < itemRating.item;
    }

    bool operator>(const ItemRating &itemRating) const {
        return rating > itemRating.rating;
    }
} ItemRating;

// for sorting similarity vector

typedef struct Similarity {
    unsigned int userId;
    float similarityValue;

    bool operator>(const Similarity &similarity) const {
        return similarityValue > similarity.similarityValue;
    }
} Similarity;

typedef struct {
    vector<float> data;
    vector<unsigned int> cols;
    vector<unsigned int> rowPtrs;
    vector<float> userEuclideanNorm;
    vector<float> userMean;
} RatingsMatrixCSR;

RatingsMatrixCSR *readInputRatings(string &file, char *delim);

void displayRatingMatrix(RatingsMatrixCSR &ratingMatrix);

void normalizeRatingVectors(RatingsMatrixCSR &ratingsMatrix);

typedef struct SimilarityMatrix {
    float *similarities;
    unsigned int size;
} SimilarityMatrix;

void initSimilarityMatrix(SimilarityMatrix &similarityMatrix);

void displaySimilarityMatrix(SimilarityMatrix &similarityMatrix);

void displayRecommendations(vector<ItemRating> &recommendations, map<unsigned int, string> &movieIdNameMapping);

map<unsigned int, string> readInputMovies(string &file, char *delim);

vector<unsigned int> readInputUserIds(string &file);

vector<unsigned int> getMovieIds(map<unsigned int, string> &movieIdNameMapping);

bool verifySimilarityMatrix(SimilarityMatrix &goldMatrix, SimilarityMatrix &kernelMatrix);

#endif /* RATINGS_UTIL_H */
