#include <iostream>
#include <stdlib.h>

#include "ratings_util.h"
#include "similarity_gold.cpp"

int main(int argc, char* argv[]) {

    //read input and construct user rating matrix
    const char * fileName = "data/ratings.csv";//TODO read from args
    RatingsMatrixCSR * rating_matrix = readInputRatings(fileName);    
    displayRatingMatrix(rating_matrix);
    
    //compute similarity in sequential version (gold)
    computeSimilarity(rating_matrix);    

    //compute similarity in parallel version (kernel)
}

