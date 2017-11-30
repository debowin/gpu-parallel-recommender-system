#include "similarity_gold.h"
#include "misc_utls.h"


int main(int argc, char *argv[]) {
    // read input and construct user rating matrix
    string fileName;
    if (argc != 2) {
        cout << "Usage: parallel-recommenders <path-to-rating-csv>" << endl;
        exit(1);
    }
    fileName = argv[1];
    RatingsMatrixCSR *ratingMatrix = readInputRatings(fileName);

    normalizeRatingVectors(*ratingMatrix);

    displayRatingMatrix(*ratingMatrix);

    //compute similarity in sequential version (gold)
    Timer timer{};
    cout << "Computing UU Similarity - Gold" << endl;
    startTime(&timer);
    SimilarityMatrix similarityMatrix = computeSimilarity(*ratingMatrix);
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));
//    displaySimilarityMatrix(similarityMatrix);

    //compute similarity in parallel version (kernel)
}

