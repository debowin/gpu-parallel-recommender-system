#include "recommendations_gold.h"
#include "misc_utls.h"


int main(int argc, char *argv[]) {
    // read input and construct user rating matrix
    if (argc != 6) {
        cout << "Usage: parallel-recommenders <path-to-rating-file> <path-to-movie-file> "
                "<delimiter> <path-to-input-user-ids> <n>" << endl;
        exit(1);
    }
    string ratingsFileName = argv[1];
    string moviesFileName = argv[2];
    char *delim = argv[3];
    string userIdsFileName = argv[4];
    auto n = (unsigned int) strtol(argv[5], nullptr, 10);

    RatingsMatrixCSR *ratingMatrix;
    try {
        ratingMatrix = readInputRatings(ratingsFileName, delim);
    } catch (invalid_argument &err) {
        cout << err.what() << ratingsFileName << endl;
        exit(2);
    }

    map<unsigned int, string> movieIdNameMapping;
    vector<unsigned int> movieIds;
    try {
        movieIdNameMapping = readInputMovies(moviesFileName, delim);
    } catch (invalid_argument &err) {
        cout << err.what() << ratingsFileName << endl;
        exit(2);
    }

    vector<unsigned int> inputUserIds;
    try {
        inputUserIds = readInputUserIds(userIdsFileName);
    } catch (invalid_argument &err) {
        cout << err.what() << userIdsFileName << endl;
        exit(2);
    }

    movieIds = getMovieIds(movieIdNameMapping);

    Timer timer{};
    startTime(&timer);
    normalizeRatingVectors(*ratingMatrix);
    stopTime(&timer);
    cout << "Normalizing Ratings took " << elapsedTime(timer) << " seconds." << endl;

    // compute similarity in sequential version (gold)
    cout << "Computing UU Similarity - Gold" << endl;
    startTime(&timer);
    SimilarityMatrix similarityMatrix = computeSimilarityGold(*ratingMatrix);
    stopTime(&timer);
    cout << endl << "Took " << setprecision(6) << elapsedTime(timer) << " seconds." << endl;

    // recommend top n for given user ids in sequential version (gold)
    cout << endl << "Calculating Top-" << n << " Recommendations for " << inputUserIds.size() << endl;
    startTime(&timer);
    for (unsigned int inputUserId : inputUserIds) {
        cout << endl << "User #" << inputUserId << endl;
        vector<ItemRating> recommendations = calculateTopNRecommendationsForUserGold(*ratingMatrix,
                                                                                     similarityMatrix,
                                                                                     movieIds,
                                                                                     inputUserId - 1, n);
        displayRecommendations(recommendations, movieIdNameMapping);
    }
    stopTime(&timer);
    cout << endl << "Took " << setprecision(6) << elapsedTime(timer)
         << " seconds for " << inputUserIds.size() << " users." << endl;

    // TODO compute similarity in parallel version (kernel)

    free(similarityMatrix.similarities);
    free(ratingMatrix);
}

