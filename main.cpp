#include "recommendations_gold.h"
#include "misc_utls.h"


int main(int argc, char *argv[]) {
    // read input and construct user rating matrix
    string ratingsFileName, moviesFileName;
    unsigned int userId, n;
    if (argc != 5) {
        cout << "Usage: parallel-recommenders <path-to-rating-csv> <path-to-movie-csv> <user-id> <n>" << endl;
        exit(1);
    }
    ratingsFileName = argv[1];
    moviesFileName = argv[2];
    userId = (unsigned int) strtol(argv[3], nullptr, 10);
    n = (unsigned int) strtol(argv[4], nullptr, 10);

    RatingsMatrixCSR *ratingMatrix;
    try {
        ratingMatrix = readInputRatings(ratingsFileName);
    } catch (invalid_argument &err) {
        cout << err.what() << ratingsFileName << endl;
        exit(2);
    }

    map<unsigned int, string> movieIdNameMapping;
    vector<unsigned int> movieIds;
    try {
        movieIdNameMapping = readInputMovies(moviesFileName);
    } catch (invalid_argument &err) {
        cout << err.what() << ratingsFileName << endl;
        exit(2);
    }

    movieIds = getMovieIds(movieIdNameMapping);

    normalizeRatingVectors(*ratingMatrix);

    Timer timer{};
    // compute similarity in sequential version (gold)
    cout << "Computing UU Similarity - Gold" << endl;
    startTime(&timer);
    SimilarityMatrix similarityMatrix = computeSimilarityGold(*ratingMatrix);
    stopTime(&timer);
    cout << endl << "Took " << setprecision(6) << elapsedTime(timer) << " seconds." << endl;

    // recommend top n in sequential version (gold)
    cout << endl << "Calculating Top-" << n << " Recommendations for User #" << userId << endl;
    startTime(&timer);
    vector<ItemRating> recommendations = calculateTopNRecommendationsForUserGold(*ratingMatrix,
                                                                                 similarityMatrix,
                                                                                 movieIds,
                                                                                 userId - 1, n);
    stopTime(&timer);
    displayRecommendations(recommendations, movieIdNameMapping);
    cout << endl << "Took " << setprecision(6) << elapsedTime(timer) << " seconds." << endl;

    // TODO compute similarity in parallel version (kernel)
}

