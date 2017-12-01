#include "similarity_gold.h"
#include "misc_utls.h"


int main(int argc, char *argv[]) {
    // read input and construct user rating matrix
    string fileName;
    unsigned int userId;
    if (argc != 3) {
        cout << "Usage: parallel-recommenders <path-to-rating-csv>" << endl;
        exit(1);
    }
    fileName = argv[1];
    userId = (unsigned int) strtol(argv[2], nullptr, 10);
    RatingsMatrixCSR *ratingMatrix = readInputRatings(fileName);

    normalizeRatingVectors(*ratingMatrix);

    displayRatingMatrix(*ratingMatrix);

    // compute similarity in sequential version (gold)
    Timer timer{};
    cout << "Computing UU Similarity - Gold" << endl;
    startTime(&timer);
    SimilarityMatrix similarityMatrix = computeSimilarity(*ratingMatrix);
    stopTime(&timer);
    cout << fixed << setprecision(6) << elapsedTime(timer) << " seconds." << endl;
    displaySimilarityMatrix(similarityMatrix);

    // compute similarity in parallel version (kernel)
}

