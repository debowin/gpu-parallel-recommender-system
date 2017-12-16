#include <cmath>
#include <algorithm>
#include "recommendations_gold.h"

SimilarityMatrix computeSimilarityGold(RatingsMatrixCSR &ratingMatrix) {
    SimilarityMatrix similarityMatrix = {nullptr, (unsigned int) ratingMatrix.rowPtrs.size() - 1};
    initSimilarityMatrix(similarityMatrix);

    for (unsigned int i = 0; i < ratingMatrix.rowPtrs.size() - 2; i++) {
        // for each user
        for (unsigned int j = i + 1; j < ratingMatrix.rowPtrs.size() - 1; j++) {
            // for every other user
            float similarity = calculatePearsonCorrelationGold(ratingMatrix, i, j);
            similarityMatrix.similarities[i * similarityMatrix.size + j] = similarity;
            similarityMatrix.similarities[j * similarityMatrix.size + i] = similarity;
        }
    }
    return similarityMatrix;
}

float calculatePearsonCorrelationGold(RatingsMatrixCSR &ratingsMatrix, unsigned int user1, unsigned int user2) {
    int end1 = ratingsMatrix.rowPtrs[user1 + 1], end2 = ratingsMatrix.rowPtrs[user2 + 1];
    int index1 = ratingsMatrix.rowPtrs[user1], index2 = ratingsMatrix.rowPtrs[user2];
    float similarity = 0.f;
    float user1Norm = ratingsMatrix.userEuclideanNorm[user1];
    float user2Norm = ratingsMatrix.userEuclideanNorm[user2];
    if(!user1Norm || !user2Norm)
        return 0.0f;
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
    return similarity / (user1Norm * user2Norm);
}

vector<ItemRating> calculateTopNRecommendationsForUserGold(RatingsMatrixCSR &ratingsMatrix,
                                                           SimilarityMatrix &similarityMatrix,
                                                           vector<unsigned int> &movieIds,
                                                           unsigned int userId, unsigned int n) {
    vector<ItemRating> recommendations;
    unsigned int item = ratingsMatrix.rowPtrs[userId];
    unsigned int end = ratingsMatrix.rowPtrs[userId + 1];
    // find the unrated items for this user
    for (auto &movieId : movieIds) {
        if (item >= end || movieId < ratingsMatrix.cols[item])
            recommendations.push_back(ItemRating{movieId, 0});
        else if (movieId == ratingsMatrix.cols[item])
            item += 1;
    }

    // order neighbors by similarity
    vector<Similarity> sortedSimilarities;
    for (unsigned int i = 0; i < similarityMatrix.size; i++) {
        if (similarityMatrix.similarities[userId * similarityMatrix.size + i] <= 0)
            // ignore any similarity that's not positive
            continue;
        sortedSimilarities.push_back(
                Similarity{i, similarityMatrix.similarities[userId * similarityMatrix.size + i]});
    }
    sort(sortedSimilarities.begin(), sortedSimilarities.end(), greater<Similarity>());

    unsigned int k = 20;

    for (ItemRating &itemRating: recommendations) {
        // for each unrated item
        unsigned int neighborCount = 0;
        float predictedRating = 0.f;
        float similaritySum = 0.f;
        // find top k neighbors who have rated item
        for (auto &sortedSimilarity : sortedSimilarities) {
            // binary search for item in neighbor's rating vector
            int index = binaryLocate(ratingsMatrix.cols, ratingsMatrix.rowPtrs[sortedSimilarity.userId],
                                     ratingsMatrix.rowPtrs[sortedSimilarity.userId + 1] - 1, itemRating.item);
            if (index == -1)
                // if neighbor hasn't rated the item
                continue;
            predictedRating += sortedSimilarity.similarityValue * ratingsMatrix.data[index];
            similaritySum += sortedSimilarity.similarityValue;
            neighborCount += 1;
            // we only need k neighbors
            if (neighborCount == k)
                break;
        }
        if (neighborCount > 2)
            // refuse to predict score for an item with less than 3 neighbors
            itemRating.rating = predictedRating / similaritySum + ratingsMatrix.userMean[userId];
    }

    sort(recommendations.begin(), recommendations.end(), greater<ItemRating>());

    if (recommendations.size() > n)
        recommendations.resize(n);
    return recommendations;
}

int binaryLocate(vector<unsigned int> &array, int l, int r, unsigned int target) {
    if (l > r)
        return -1;
    int mid = (l + r) / 2;
    if (target == array[mid])
        return mid;
    else if (target < array[mid])
        return binaryLocate(array, l, mid - 1, target);
    else if (target > array[mid])
        return binaryLocate(array, mid + 1, r, target);
}