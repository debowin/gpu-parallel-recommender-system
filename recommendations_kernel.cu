#include "recommendations_kernel.h"
#include <queue>
#include <algorithm>

//TODO tuning
#define BLOCK_SIZE_REC 64
#define BLOCK_SIZE_DIV 64 

void allocateDeviceMemory(void ** d_data, size_t size);
void copyToDeviceMemory(void * d_data, void * h_data, size_t size);
void copyFromDeviceMemory(void * h_data, void * d_data, size_t size);

//basic kernel function to compute UU similarity from csr rating data
__global__ void csrSimilarityKernel(unsigned int dim, unsigned int * csrRowPtr, 
                 unsigned int * csrColIdx, float * csrData, float * userEuclideanNorm, float * output) {

    // get row ids for which dot product needs to be computed 
    unsigned int row_x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row_y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((row_x >= dim) || (row_y >= dim))
        // out of bounds
        return;
    if (row_x == row_y){
        output[row_y + row_x * dim] = 0.0f;
        return;
    }
    float row_x_norm = userEuclideanNorm[row_x];
    float row_y_norm = userEuclideanNorm[row_y];
    if(!row_x_norm || !row_y_norm) {
        // if either is a zero vector
        output[row_y + row_x * dim] = 0.0f;
        return;
    }

    unsigned int id_x = csrRowPtr[row_x]; // set to start of row 1
    unsigned int id_y = csrRowPtr[row_y]; // set to start of row 2
                                
    unsigned int col_id_x, col_id_y;
    // compute similarity
    float similarity = 0.0f;
    while (id_x < csrRowPtr[row_x + 1] && id_y < csrRowPtr[row_y + 1]) {
        col_id_x = csrColIdx[id_x];
        col_id_y = csrColIdx[id_y];
        if (col_id_x == col_id_y) {
            similarity += csrData[id_x] * csrData[id_y];
            id_x ++;
            id_y ++;
        }
        else if (col_id_x < col_id_y) {
            id_x ++;
        }
        else {
            id_y ++;  
        }
    }
    output[row_y + dim * row_x] = similarity/(row_x_norm * row_y_norm);
}

// coarsened kernel where one thread writes to two result locations.
__global__ void csrSimilarityKernelCoarsened(unsigned int dim, unsigned int * csrRowPtr,
                 unsigned int * csrColIdx, float * csrData, float * userEuclideanNorm, float * output){
    unsigned int t_id = threadIdx.x + blockIdx.x * blockDim.x;
    // check out of bounds
    if(t_id >= ((dim-1)*dim)/2)
        return;
    // determine row ids for which dot product needs to be computed 
    unsigned int row_x = 0;
    unsigned int row_y = 0;
    unsigned int subtractor = dim - 1;
    while(t_id >= subtractor) {
        row_x++;
        t_id -= subtractor;
        subtractor--;
    }
    row_y = row_x + t_id + 1;
    float row_x_norm = userEuclideanNorm[row_x];
    float row_y_norm = userEuclideanNorm[row_y];
    // if either is a zero vector
    if(!row_x_norm || !row_y_norm) {
        output[row_y + row_x * dim] = 0.0f;
        return;
    }
    unsigned int id_x = csrRowPtr[row_x]; // set to start of row 1
    unsigned int id_y = csrRowPtr[row_y]; // set to start of row 2
                                
    unsigned int col_id_x, col_id_y;
    // compute similarity
    float similarity = 0.0f;
    while (id_x < csrRowPtr[row_x + 1] && id_y < csrRowPtr[row_y + 1]) {
        col_id_x = csrColIdx[id_x];
        col_id_y = csrColIdx[id_y];
        if (col_id_x == col_id_y) {
            similarity += csrData[id_x] * csrData[id_y];
            id_x ++;
            id_y ++;
        }
        else if (col_id_x < col_id_y) {
            id_x ++;
        }
        else {
            id_y ++;  
        }
    }
    output[row_y + dim * row_x] = similarity/(row_x_norm * row_y_norm);
    output[row_x + dim * row_y] = similarity/(row_x_norm * row_y_norm);
}

//shared implementation of csr similarity kernel
__global__ void csrSimilarityKernelShared(unsigned int dim, unsigned int * csrRowPtr,
                 unsigned int * csrColIdx, float * csrData, float * userEuclideanNorm, float * output) {

    // row info
    __shared__ unsigned int row_start_x_sh; // start index
    __shared__ unsigned int row_end_x_sh; // end index 
    __shared__ float data_x[TILE_SIZE]; // row data 
    __shared__ unsigned int cols_x[TILE_SIZE];// col ids
    __shared__ float row_x_norm_sh; //euclidean norm

    unsigned int tid = threadIdx.x;
    unsigned int row_x = blockIdx.x;
    
    // initialize row ptrs
    if (tid == 0) {
        row_start_x_sh = csrRowPtr[row_x];
        row_end_x_sh = csrRowPtr[row_x + 1];
        row_x_norm_sh = userEuclideanNorm[row_x];  
    } 
    // make sure the basic row info is loaded
    __syncthreads(); 
    
    unsigned int row_start_x = row_start_x_sh;
    unsigned int row_end_x = row_end_x_sh;
    unsigned int tile_idx = tid;
    unsigned int csr_idx = tile_idx + row_start_x;
    //load csr data into shared memory
    while (csr_idx < row_end_x && tile_idx < TILE_SIZE) {
        data_x[tile_idx] = csrData[csr_idx];
        cols_x[tile_idx] = csrColIdx[csr_idx];
        tile_idx += blockDim.x;
        csr_idx += blockDim.x;
    }
    // make sure the row data and col ids are loaded
    __syncthreads();

    
    unsigned int row_y = tid + blockIdx.x + 1;
    unsigned int id_x;
    unsigned int end_x = row_end_x - row_start_x;
    unsigned int id_y;
    unsigned int end_y;
    float row_x_norm = row_x_norm_sh;
    float row_y_norm; 
    while (row_y < dim) {
       id_x = 0;
       id_y = csrRowPtr[row_y];
       end_y = csrRowPtr[row_y + 1];
       row_y_norm = userEuclideanNorm[row_y];
       if (!row_x_norm || !row_y_norm) {
           output[row_y + dim * row_x] = 0.0f;
           output[row_x + dim * row_y] = 0.0f;
           row_y += blockDim.x;
           continue;
       }
       
       unsigned int col_id_x, col_id_y;
       // compute similarity
       float similarity = 0.0f;
       while (id_x < end_x && id_y < end_y) {
           col_id_x = cols_x[id_x];
           col_id_y = csrColIdx[id_y];
           if (col_id_x == col_id_y) {
               similarity += data_x[id_x] * csrData[id_y];
               id_x ++;
               id_y ++;
           }
           else if (col_id_x < col_id_y) {
               id_x ++;
           }
           else {
               id_y ++;
           }
       }
       similarity /= (row_x_norm * row_y_norm);
       output[row_y + dim * row_x] = similarity;
       output[row_x + dim * row_y] = similarity;
       row_y += blockDim.x;
    } 
}

//kernel to do vector division to compute final prediction score
__global__ void computeFinalPredictionScores(ItemRating *recommendations, float *similarities_sum,
                                            unsigned int rec_size, float userMean)  {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float score;
    if (tid < rec_size) {
        score = recommendations[tid].rating;
        float similarity_sum = similarities_sum[tid]; 
        if (similarity_sum > 0.0f) {
            score /= similarity_sum;
            score += userMean;
            recommendations[tid].rating = score;
        }
    }
}


//fetches the item index if it exists
__device__ int getItemIndex(unsigned int * col_ids, unsigned int count, unsigned int item_id) {
    //simple search
    unsigned int col;
    for (int i = 0; i < count; i++) {
        col = col_ids[i];
        if (item_id == col) {
            return i;
        }
        else if (item_id < col) {
            return -1;
        }
    }
    return -1;
}


//kernel to compute predictions for user
__global__ void computePredictionsForUserKernel(unsigned int *csrRowPtr, unsigned int *csrColIdx, float *csrData,
                                                ItemRating *recommendations, Similarity *sortedNeighbours, float *similarities_sum,
                                                unsigned int rec_size, unsigned int neighbour_size) {
     //rating info of neighbour user
    __shared__ unsigned int row_start_sh;//start index
    __shared__ unsigned int row_end_sh;//end index
    __shared__ unsigned int row_count_sh;//total elements
    __shared__ float data[TILE_SIZE];//row data
    __shared__ unsigned int cols[TILE_SIZE];//col ids
    __shared__ Similarity neighbour;

    unsigned int tid = threadIdx.x;

    //load initial row info
    if (tid == 0) {
        neighbour = sortedNeighbours[blockIdx.x];
        unsigned int row = neighbour.userId;
        row_start_sh = csrRowPtr[row];
        row_end_sh = csrRowPtr[row + 1];
        //restricting data to TILE size  
        row_count_sh = fminf(TILE_SIZE, row_end_sh - row_start_sh);
    }
    __syncthreads();

    //load from Shared Memory to local regs
    unsigned int row_start = row_start_sh;
    unsigned int row_end = row_end_sh;
    unsigned int row_count = row_count_sh;

    unsigned int tile_idx = tid;
    unsigned int csr_idx = tile_idx + row_start;

    //load user data into shared memory
    while (csr_idx < row_end && tile_idx < TILE_SIZE) {
        data[tile_idx] = csrData[csr_idx];
        cols[tile_idx] = csrColIdx[csr_idx];
        tile_idx += blockDim.x;
        csr_idx += blockDim.x;
    }
    __syncthreads();

    unsigned int max_item_id = cols[row_count - 1];
    ItemRating item_rating;
    unsigned int item_itr = tid;
    int item_idx;
    float similarity = neighbour.similarityValue;
    float result;
    //iterate through input set of items
    while (item_itr < rec_size) {
        item_rating = recommendations[item_itr];
        //if item id exceeds maximum rated item id then return
        if (item_rating.item >  max_item_id) {
            return;
        }
        //check if item exists in user rated items (col ids)
        if ((item_idx = getItemIndex(cols, row_count, item_rating.item)) != -1) {
            result = data[item_idx] * similarity;
            //atomic add result score (numerator)
            atomicAdd(&(recommendations[item_itr].rating), result);
            //atomic add similarity sum (denominator)
            atomicAdd(&(similarities_sum[item_itr]), similarity);
        }
        item_itr += blockDim.x;
    }
}

//wrapper function to top n recs kernel
vector<ItemRating> calculateTopNRecommendationsForUserParallel(unsigned int *csrRowPtr_d, unsigned int *csrColIdx_d, float *csrData_d,
                                                 SimilarityMatrix similarityMatrix, vector<unsigned int> movieIds,
                                                 RatingsMatrixCSR &ratingsMatrix, unsigned int userId, unsigned int N) {

    //find the unrated items for this user (same as Gold)
    vector<ItemRating> recommendations;
    unsigned int item = ratingsMatrix.rowPtrs[userId];
    unsigned int end = ratingsMatrix.rowPtrs[userId + 1];
    for (auto &movieId : movieIds) {
        if (item >= end || movieId < ratingsMatrix.cols[item])
            recommendations.push_back(ItemRating{movieId, 0});
        else if (movieId == ratingsMatrix.cols[item])
            item += 1;
    }

    //add similar users into a priority queue
    priority_queue <Similarity, vector<Similarity>, greater<Similarity> > similarUsers;
    unsigned int neighbourhood_size = similarityMatrix.size/20; //considering 5% more similar users in neighbourhood
    for (unsigned int i = 0; i < similarityMatrix.size; i++) {
        
        float similarityValue = similarityMatrix.similarities[userId * similarityMatrix.size + i];
        //ignore any similarity that's not positive
        if (i == userId || similarityValue <= 0)
            continue;
        Similarity currUser = Similarity{i, similarityValue};
        if (similarUsers.size() < neighbourhood_size) {
            similarUsers.push(currUser);
        }
        else {
            if (currUser > similarUsers.top()) {
                similarUsers.pop();
                similarUsers.push(currUser);
            }
        }
    }

    ItemRating *recommendations_d;
    Similarity *similarUsers_d;
    float * similaritySum_d;

    //allocate memory
    allocateDeviceMemory((void **)&recommendations_d, sizeof(ItemRating) * recommendations.size());
    allocateDeviceMemory((void **)&similarUsers_d, sizeof(Similarity) * similarUsers.size());
    allocateDeviceMemory((void **)&similaritySum_d, sizeof(float) * recommendations.size());
    

    //copy to Device memory
    copyToDeviceMemory(recommendations_d, &recommendations[0], sizeof(ItemRating) * recommendations.size());
    copyToDeviceMemory(similarUsers_d, (Similarity *) &similarUsers.top(), sizeof(Similarity) * similarUsers.size());
    //initialize all values to 0
    cudaMemset(similaritySum_d, 0, sizeof(float) * recommendations.size());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;     

    //the kernel starts here
    unsigned int noOfBlocks = similarUsers.size();
    cudaEventRecord(start);
    computePredictionsForUserKernel<<<noOfBlocks, BLOCK_SIZE_REC>>>(csrRowPtr_d, csrColIdx_d, csrData_d, recommendations_d,
                                           similarUsers_d, similaritySum_d, recommendations.size(), similarUsers.size());
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Recommendations Kernel exec time: %f ms \n", milliseconds);    


    float userMean = ratingsMatrix.userMean[userId];
    noOfBlocks = ceil((float)recommendations.size()/BLOCK_SIZE_DIV);

    cudaEventRecord(start);
    computeFinalPredictionScores<<<noOfBlocks, BLOCK_SIZE_DIV>>>(recommendations_d, similaritySum_d, recommendations.size(), userMean);
    cudaEventRecord(stop);

    copyFromDeviceMemory(&recommendations[0], recommendations_d, sizeof(ItemRating) * recommendations.size());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Division Kernel exec time: %f ms \n", milliseconds);


    //fetch top N recommendations
    priority_queue <ItemRating, vector<ItemRating>, greater<ItemRating> > topRecommendations;
    for (int i = 0; i < recommendations.size(); i++) {
        if (topRecommendations.size() < N) {
            topRecommendations.push(recommendations[i]);
        }
        else {
            if (recommendations[i] > topRecommendations.top()) {
                topRecommendations.pop();
                topRecommendations.push(recommendations[i]);
            }
        }
    }

    vector<ItemRating> sortedTopRecommendations;
    while (!topRecommendations.empty()) {
        sortedTopRecommendations.push_back(topRecommendations.top());
        topRecommendations.pop();        
    }

    sort(sortedTopRecommendations.begin(), sortedTopRecommendations.end(), greater<ItemRating>());
    return sortedTopRecommendations;
}


//wrapper to kernel function
SimilarityMatrix computeSimilarityParallel(unsigned int dim, unsigned int *csrRowPtr_d, unsigned int *csrColIdx_d,
                                      float *csrData_d,  float *userEuclideanNorm_d) {
      
    float *output_d;
    //allocate memory for output
    allocateDeviceMemory((void **)&output_d, sizeof(float) * (dim * dim));

    //call csr kernel 1
    dim3 grid_dim, block_dim;
    block_dim.x = BLOCK_DIM_X;
    block_dim.y = BLOCK_DIM_Y;
    grid_dim.x = ceil((float)dim/BLOCK_DIM_X);
    grid_dim.y = ceil((float)dim/BLOCK_DIM_Y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f; 

    float * similarities = (float *)malloc(sizeof(float) * (dim * dim));
  
    ////BASIC KERNEL////
 
    cudaEventRecord(start);
    csrSimilarityKernel<<<grid_dim, block_dim>>>(dim, csrRowPtr_d, csrColIdx_d, csrData_d, userEuclideanNorm_d, output_d);
    cudaEventRecord(stop);

    // //display results of kernel 1
    copyFromDeviceMemory(similarities, output_d, sizeof(float) * (dim * dim));
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Basic Kernel time: %f ms \n",  milliseconds);

    ////COARSENED KERNEL////

    // unsigned int bestThreadCount = ((dim-1)*dim)/2;
    // cudaEventRecord(start);
    // csrSimilarityKernelCoarsened<<<ceil(bestThreadCount/1024.0f), 1024>>>(dim, csrRowPtr_d, csrColIdx_d, csrData_d, userEuclideanNorm_d, output_d);
    // cudaEventRecord(stop);

    // //display results of kernel 1
    // copyFromDeviceMemory(similarities, output_d, sizeof(float) * (dim * dim));
    // cudaEventSynchronize(stop);

    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Coarsened Kernel time: %f ms \n",  milliseconds);
    
    ////SHARED KERNEL////

    //cudaEventRecord(start);
    //csrSimilarityKernelShared<<<dim, BLOCK_SIZE>>>(dim, csrRowPtr_d, csrColIdx_d, csrData_d, userEuclideanNorm_d, output_d);
    //cudaEventRecord(stop);

    //copyFromDeviceMemory(similarities, output_d, sizeof(float) * (dim * dim));
    //cudaEventSynchronize(stop);

    //cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("Shared Kernel time: %f ms \n", milliseconds);

    SimilarityMatrix outputSimilarityMatrix = {similarities, dim};
    return outputSimilarityMatrix;
}

void allocateMemoryToDevicePtrs(unsigned int dim, unsigned int **csrRowPtr_d, unsigned int **csrColIdx_d,
                                float **csrData_d, float **userEuclideanNorm_d,  RatingsMatrixCSR &ratingMatrix) {
    //allocate memory for row ptr
    allocateDeviceMemory((void **)csrRowPtr_d, sizeof(unsigned int) * (dim + 1));
    //allocate memory for col ids
    allocateDeviceMemory((void **)csrColIdx_d, sizeof(unsigned int) * ratingMatrix.cols.size());
    //allocate memory for normalized ratings data
    allocateDeviceMemory((void **)csrData_d, sizeof(float) * ratingMatrix.data.size());
    //allocate memory for user euclidien distance
    allocateDeviceMemory((void **)userEuclideanNorm_d, sizeof(float) * ratingMatrix.userEuclideanNorm.size());
}

void copyRatingsMatrixToDevicePtrs(unsigned int dim, unsigned int *csrRowPtr_d, unsigned int *csrColIdx_d,
                                   float *csrData_d, float *userEuclideanNorm_d, RatingsMatrixCSR &ratingMatrix) {

    //copy row ptr to Device Memory
    unsigned int * csrRowPtr = &ratingMatrix.rowPtrs[0];
    copyToDeviceMemory(csrRowPtr_d, csrRowPtr, sizeof(unsigned int) * (dim + 1));
    //copy cold ids to Device Memory
    unsigned int * csrColIdx =  &ratingMatrix.cols[0];
    copyToDeviceMemory(csrColIdx_d, csrColIdx, sizeof(unsigned int) * ratingMatrix.cols.size());
    //copy data to Device Memory
    float * csrData = &ratingMatrix.data[0];
    copyToDeviceMemory(csrData_d, csrData, sizeof(float) * ratingMatrix.data.size());
    //copy euclidean norm to device memory
    float * userEuclideanNorm = &ratingMatrix.userEuclideanNorm[0];
    copyToDeviceMemory(userEuclideanNorm_d, userEuclideanNorm, sizeof(float) * ratingMatrix.userEuclideanNorm.size());
}


void allocateDeviceMemory(void ** d_data, size_t size) 
{
    cudaError_t cuda_ret;
    cuda_ret = cudaMalloc(d_data, size);
    if(cuda_ret != cudaSuccess) 
        printf("Unable to allocate device memory");    
}

void copyToDeviceMemory(void * d_data, void * h_data, size_t size) 
{
    cudaError_t cuda_ret;
    cuda_ret = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) 
        fprintf(stderr, "Unable to copy to device memory");
}

void copyFromDeviceMemory(void * h_data, void * d_data, size_t size)
{
    cudaError_t cuda_ret;
    cuda_ret = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess)
        fprintf(stderr, "Unable to copy from device memory");
}

