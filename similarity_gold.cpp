#include <cmath>
#include "similarity_gold.h"

SimilarityMatrix computeSimilarity(RatingsMatrixCSR &ratingMatrix) {
    SimilarityMatrix similarityMatrix = {nullptr, (unsigned int) ratingMatrix.rowPtrs.size() - 1,
                                         (unsigned int) ratingMatrix.rowPtrs.size() - 1};
    initSimilarityMatrix(similarityMatrix);

    for (unsigned int i = 0; i < ratingMatrix.rowPtrs.size() - 2; i++) {
        // for each user
        for (unsigned int j = i + 1; j < ratingMatrix.rowPtrs.size() - 1; j++) {
            // for every other user
            float similarity = calculatePearsonCorrelation(ratingMatrix, i, j);
            similarityMatrix.similarities[i * similarityMatrix.width + j] = similarity;
            similarityMatrix.similarities[j * similarityMatrix.width + i] = similarity;
        }
    }
    return similarityMatrix;
}

float calculatePearsonCorrelation(RatingsMatrixCSR &ratingsMatrix, unsigned int user1, unsigned int user2) {
    int end1 = ratingsMatrix.rowPtrs[user1 + 1], end2 = ratingsMatrix.rowPtrs[user2 + 1];
    int index1 = ratingsMatrix.rowPtrs[user1], index2 = ratingsMatrix.rowPtrs[user2];
    float similarity = 0.f;
    while (index1 < end1 && index2 < end2) {
        if (ratingsMatrix.cols[index1] == ratingsMatrix.cols[index2]) {
            similarity += ratingsMatrix.data[index1] * ratingsMatrix.data[index2];
            index1 += 1;
            index2 += 1;
        } else if (ratingsMatrix.cols[index1] < ratingsMatrix.cols[index2])
            index1 += 1;
        else
            index2 += 1;
    }
    return similarity / (ratingsMatrix.userEuclideanNorm[user1] * ratingsMatrix.userEuclideanNorm[user2]);
}
