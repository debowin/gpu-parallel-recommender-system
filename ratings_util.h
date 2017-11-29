#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <vector>

using namespace std;

typedef struct RatingsMatrixCSR {
    vector<float> data;
    vector<unsigned int> cols;
    vector<unsigned int> row_ptr;
} RatingsMatrixCSR;

RatingsMatrixCSR * readInputRatings(const char * file);

void displayRatingMatrix(RatingsMatrixCSR * rating_matrix); 
