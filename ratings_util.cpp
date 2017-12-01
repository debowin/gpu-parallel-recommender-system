#include <algorithm>
#include "ratings_util.h"

RatingsMatrixCSR *readInputRatings(string file) {

    ifstream ratingsFile;
    //open the ratings file
    ratingsFile.open(file);

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

//        ratingsMatrix->cols.push_back(itemId);

        //update rating
        char *ratingString = strtok(nullptr, ",");
        auto rating = strtof(ratingString, nullptr);
//        ratingsMatrix->data.push_back();
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
    similarityMatrix.similarities = (float *) malloc(sizeof(float) * similarityMatrix.length * similarityMatrix.width);
    memset(similarityMatrix.similarities, 0, sizeof(float) * similarityMatrix.length * similarityMatrix.width);
}

void displaySimilarityMatrix(SimilarityMatrix &similarityMatrix) {
    for (unsigned int i = 0; i < similarityMatrix.length; i++) {
        for (unsigned int j = 0; j < similarityMatrix.width; j++) {
            unsigned int index = i * similarityMatrix.width + j;
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