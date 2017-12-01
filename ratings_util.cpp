#include <algorithm>
#include "ratings_util.h"

RatingsMatrixCSR *readInputRatings(string &file) {

    ifstream ratingsFile;
    // open the ratings file
    ratingsFile.open(file);
    if (ratingsFile.fail())
        throw invalid_argument("Unable to open ratings file ");

    string line;
    unsigned int currUserId = 0;
    unsigned int totalRatings = 0;

    auto *ratingsMatrix = (RatingsMatrixCSR *) malloc(sizeof(RatingsMatrixCSR));

    ratingsFile >> line; // header

    vector<ItemRating> itemRatings;

    while (ratingsFile >> line) {
        // update row ptr
        char *userId = strtok((char *) line.c_str(), ",\n");
        if (currUserId != strtol(userId, nullptr, 10)) {
            ratingsMatrix->rowPtrs.push_back(totalRatings);
            currUserId = (unsigned int) strtol(userId, nullptr, 10);
            // sort and add previous users' item-wise ratings
            sort(itemRatings.begin(), itemRatings.end());
            for (ItemRating itemRating: itemRatings) {
                ratingsMatrix->data.push_back(itemRating.rating);
                ratingsMatrix->cols.push_back(itemRating.item);
            }
            itemRatings.clear();
        }

        //update col idx 
        char *itemIdString = strtok(nullptr, ",");
        auto itemId = (unsigned int) strtol(itemIdString, nullptr, 10);

        //update rating
        char *ratingString = strtok(nullptr, ",");
        auto rating = strtof(ratingString, nullptr);
        itemRatings.push_back(ItemRating{itemId, rating});

        //increment ratings
        totalRatings++;
    }
    ratingsMatrix->rowPtrs.push_back(totalRatings);
    // sort and add last users' item-wise ratings
    sort(itemRatings.begin(), itemRatings.end());
    for (ItemRating itemRating: itemRatings) {
        ratingsMatrix->data.push_back(itemRating.rating);
        ratingsMatrix->cols.push_back(itemRating.item);
    }
    itemRatings.clear();
    return ratingsMatrix;
}


map<unsigned int, string> readInputMovies(string &file) {
    ifstream moviesFile;
    map<unsigned int, string> movieIdNameMapping;
    // open the ratings file
    moviesFile.open(file);
    if (moviesFile.fail())
        throw invalid_argument("Unable to open movies file ");
    string line;

    getline(moviesFile, line); // header
    while (getline(moviesFile, line)) {
        char *movieIdString = strtok((char *) line.c_str(), ",\n");
        auto movieId = (unsigned int) strtol(movieIdString, nullptr, 10);
        char *movieTitle;
        if (line.find('"') != string::npos) {
            // if the name contains quotes
            movieTitle = strtok(nullptr, "\"");
        } else
            movieTitle = strtok(nullptr, ",\n");
        movieIdNameMapping[movieId] = movieTitle;
    }
    return movieIdNameMapping;
}


void displayRatingMatrix(RatingsMatrixCSR &ratingMatrix) {
    //data
    cout << "Data:" << endl;
    for (float i : ratingMatrix.data) {
        cout << i << " ";
    }
    cout << endl;
    //cols
    cout << "Cols:" << endl;
    for (unsigned int col : ratingMatrix.cols) {
        cout << col << " ";
    }
    cout << endl;
    //rows
    cout << "Row Ptr:" << endl;
    for (unsigned int i : ratingMatrix.rowPtrs) {
        cout << i << " ";
    }
    cout << endl;
}

void initSimilarityMatrix(SimilarityMatrix &similarityMatrix) {
    similarityMatrix.similarities = (float *) malloc(sizeof(float) * similarityMatrix.size * similarityMatrix.size);
    memset(similarityMatrix.similarities, 0, sizeof(float) * similarityMatrix.size * similarityMatrix.size);
}

void displaySimilarityMatrix(SimilarityMatrix &similarityMatrix) {
    for (unsigned int i = 0; i < similarityMatrix.size; i++) {
        for (unsigned int j = 0; j < similarityMatrix.size; j++) {
            unsigned int index = i * similarityMatrix.size + j;
            cout << fixed << setprecision(3) << similarityMatrix.similarities[index] << "\t";
        }
        cout << endl;
    }
}

void normalizeRatingVectors(RatingsMatrixCSR &ratingsMatrix) {
    for (unsigned int i = 0; i < ratingsMatrix.rowPtrs.size() - 1; i++) {
        // for each row, normalize it by subtracting average
        float mean = 0.f;
        for (unsigned int j = ratingsMatrix.rowPtrs[i]; j < ratingsMatrix.rowPtrs[i + 1]; j++) {
            mean += ratingsMatrix.data[j];
        }
        mean /= ratingsMatrix.rowPtrs[i + 1] - ratingsMatrix.rowPtrs[i];

        float sumOfSquares = 0.f;
        for (unsigned int j = ratingsMatrix.rowPtrs[i]; j < ratingsMatrix.rowPtrs[i + 1]; j++) {
            ratingsMatrix.data[j] -= mean;
            sumOfSquares += powf(ratingsMatrix.data[j], 2);
        }
        // pre-compute and store mean and norm
        ratingsMatrix.userMean.push_back(mean);
        ratingsMatrix.userEuclideanNorm.push_back(sqrtf(sumOfSquares));
    }
}

void displayRecommendations(vector<ItemRating> &recommendations, map<unsigned int, string> &movieIdNameMapping) {
    for (ItemRating recommendation: recommendations) {
        cout << left << setw(10) << setfill(' ')
             << fixed << recommendation.item
             << left << setw(70) << setfill(' ')
             << fixed << movieIdNameMapping[recommendation.item]
             << right << setw(10) << setfill(' ')
             << fixed << setprecision(3) << recommendation.rating << endl;
    }
}

vector<unsigned int> getMovieIds(map<unsigned int, string> &movieIdNameMapping) {
    vector<unsigned int> movieIds;
    for (auto &iter : movieIdNameMapping)
        movieIds.push_back(iter.first);
    sort(movieIds.begin(), movieIds.end());
    return movieIds;
}