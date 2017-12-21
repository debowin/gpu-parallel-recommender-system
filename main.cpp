#include "recommendations_gold.h"
#include "recommendations_kernel.h"
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
    float totalGold = 0.0;
    float totalKernel = 0.0;
    float simTime = 0.0;
    float simTime2 = 0.0;
    float recTime = 0.0;
    float recTime2 = 0.0;
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
    simTime = elapsedTime(timer);
    cout << endl << "Took " << setprecision(6) << simTime << " seconds." << endl;
    totalGold += simTime;

    // recommend top n for given user ids in sequential version (gold)
    cout << endl << "Calculating Top-" << n << " Recommendations for " << inputUserIds.size() << " users - Gold." << endl;
    vector<vector<ItemRating>> userRecommendations;
    startTime(&timer);
    for (unsigned int inputUserId : inputUserIds) {
        vector<ItemRating> recommendations = calculateTopNRecommendationsForUserGold(*ratingMatrix,
                                                                                     similarityMatrix,
                                                                                     movieIds,
                                                                                     inputUserId - 1, n);
        userRecommendations.push_back(recommendations);
    }
    stopTime(&timer);
    recTime = elapsedTime(timer);
    for (unsigned int i = 0; i < inputUserIds.size(); i++)
        storeRecommendationsToFile(userRecommendations[i], movieIdNameMapping,
                                   "goldRecommendations_" + to_string(inputUserIds[i]) + ".csv");
    
    cout << endl << "Took " << setprecision(6) << recTime
         << " seconds for " << inputUserIds.size() << " users." << endl;
    totalGold +=recTime;
    // Compute similarity in parallel version (kernel)
    startTime(&timer);
    cout << endl << "Computing UU Similarity - Parallel" << endl;

    //define CUDA device memory ptrs
    unsigned int *csrRowPtr_d;
    unsigned int *csrColIdx_d;
    float *csrData_d;
    float *userEuclideanNorm_d;
    unsigned int dim = ratingMatrix->rowPtrs.size() - 1;

    //allocate
    allocateMemoryToDevicePtrs(dim, &csrRowPtr_d, &csrColIdx_d, &csrData_d, &userEuclideanNorm_d, *ratingMatrix);
    //copy
    copyRatingsMatrixToDevicePtrs(dim, csrRowPtr_d, csrColIdx_d, csrData_d, userEuclideanNorm_d, *ratingMatrix);
    //similarities
    SimilarityMatrix similarityMatrixKernel = computeSimilarityParallel(dim, csrRowPtr_d, csrColIdx_d, csrData_d, userEuclideanNorm_d);
    
    stopTime(&timer);
    simTime2 = elapsedTime(timer);
    cout << endl << "Took " << setprecision(6) << simTime << " seconds." << endl;
    totalKernel +=simTime2;
    // Compute recommendations in parallel (kernel)

    cout << "Similarity Kernel Result Verification: "
         << (verifySimilarityMatrix(similarityMatrix, similarityMatrixKernel) ? "SUCCESS" : "FAILURE") << endl;

    cout << endl << "Calculating Top-" << n << " Recommendations for " << inputUserIds.size() << " users - Parallel." << endl;
    vector<vector<ItemRating>> userRecommendationsKernel;
    startTime(&timer);
    for (unsigned int inputUserId : inputUserIds) {
        vector<ItemRating> recommendations =  calculateTopNRecommendationsForUserParallel(csrRowPtr_d, csrColIdx_d, csrData_d,
                                                                similarityMatrixKernel, movieIds, *ratingMatrix, inputUserId - 1, n);
        userRecommendationsKernel.push_back(recommendations);
    }
    stopTime(&timer);
    recTime2 = elapsedTime(timer);
    for (unsigned int i = 0; i < inputUserIds.size(); i++)
        storeRecommendationsToFile(userRecommendationsKernel[i], movieIdNameMapping,
                                   "kernelRecommendations_" + to_string(inputUserIds[i]) + ".csv");
    cout << endl << "Took " << setprecision(6) << recTime  << " seconds for " << inputUserIds.size() << " users." << endl;

    totalKernel += recTime2;
    printf("\nTotal Kernel Time: %f\n\n", totalKernel);
    printf("\n Similarity Speedup: %f\n\n",(simTime/simTime2));
    printf("\n Recommendation Speedup: %f\n\n",(recTime/recTime2));
    printf("\n Total Speedup: %f\n\n",(totalGold/totalKernel));

    cout << "Recommendations Kernel Result Verification: "
         << (verifyRecommendations(userRecommendations, userRecommendationsKernel) ? "SUCCESS" : "FAILURE") << endl;

    // free all memories
    freeDevicePtrs(csrRowPtr_d, csrColIdx_d, csrData_d, userEuclideanNorm_d);
    free(similarityMatrix.similarities);
    free(similarityMatrixKernel.similarities);
    free(ratingMatrix);
}
