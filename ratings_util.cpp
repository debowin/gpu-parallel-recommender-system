#include "ratings_util.h"

RatingsMatrixCSR * readInputRatings(const char * file) {
   
    ifstream ratings_file;
    //open the ratigns file
    ratings_file.open(file);

    string line;
    int curr_userid = -1;
    unsigned int total_ratings = 0;    

    RatingsMatrixCSR * ratings_matrix = (RatingsMatrixCSR *) malloc(sizeof(RatingsMatrixCSR));

    ratings_file >> line;//header
    while (ratings_file >> line) {
        
        //update row ptr
        char * userid = strtok((char *)line.c_str(), ",\n");
        if (curr_userid != atoi(userid)) {
            ratings_matrix->row_ptr.push_back(total_ratings);
            curr_userid = atoi(userid);  
        }
       
        //update col idx 
        char * itemid = strtok(NULL, ",");
        unsigned int col_idx = atoi(itemid) - 1;
        ratings_matrix->cols.push_back(col_idx);
                 
        //update rating
        char * rating = strtok(NULL, ",");
        ratings_matrix->data.push_back(atof(rating));

        //increment ratings
        total_ratings++;
    }
    ratings_matrix->row_ptr.push_back(total_ratings);
    return ratings_matrix;
}


void displayRatingMatrix(RatingsMatrixCSR * rating_matrix) {
    //data
    cout << "Data:" << endl;  
    for (int i = 0; i < rating_matrix->data.size(); i++) {
        cout << rating_matrix->data[i] << " ";
    }
    cout << endl;     
    //cols
    cout << "Cols:" << endl;
    for (int i = 0; i < rating_matrix->cols.size(); i++) {
        cout << rating_matrix->cols[i] << " ";
    }
    cout << endl;
    //rows
    cout << "Row ptr:" << endl;
    for (int i = 0; i < rating_matrix->row_ptr.size(); i++) {
        cout << rating_matrix->row_ptr[i] << " ";
    }
    cout << endl;
}
