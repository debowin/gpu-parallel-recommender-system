#include "recommendations_kernel.h"

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

__global__ void csrSimilarityKernelShared(unsigned int dim, unsigned int * csrRowPtr,
                 unsigned int * csrColIdx, float * csrData, float * userEuclideanNorm, float * output) {

    // row info
    __shared__ unsigned int row_start_x; // start index
    __shared__ unsigned int row_end_x; // end index 
    __shared__ float data_x[TILE_SIZE]; // row data 
    __shared__ unsigned int cols_x[TILE_SIZE];// col ids
    __shared__ float row_x_norm; // euclidean norm

    unsigned int tid = threadIdx.x;
    unsigned int row_x = blockIdx.x;
    // initialize row ptrs
    if (tid == 0) {
        row_start_x = csrRowPtr[row_x];
        row_end_x = csrRowPtr[row_x + 1];
        row_x_norm = userEuclideanNorm[row_x];  
    } 
    // make sure the basic row info is loaded
    __syncthreads(); 
    
    // load data into shared memory
    unsigned int tile_idx = tid;
    unsigned int csr_idx = tile_idx + row_start_x;
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


//wrapper to kernel function
SimilarityMatrix computeSimilarityParallel(RatingsMatrixCSR &ratingMatrix) {
   
    unsigned int dim = ratingMatrix.rowPtrs.size() - 1;
    SimilarityMatrix similarityMatrix = {nullptr, (unsigned int) dim};
    //device DS
    unsigned int *csrRowPtr_d;
    unsigned int *csrColIdx_d;
    float *csrData_d;
    float *userEuclideanNorm_d;
    float *output_d;    
 
    //allocate memory for row ptr
    allocateDeviceMemory((void **)&csrRowPtr_d, sizeof(unsigned int) * (dim + 1));
    //allocate memory for col ids
    allocateDeviceMemory((void **)&csrColIdx_d, sizeof(unsigned int) * ratingMatrix.cols.size());
    //allocate memory for normalized ratings data
    allocateDeviceMemory((void **)&csrData_d, sizeof(float) * ratingMatrix.data.size());
    //allocate memory for user euclidien distance
    allocateDeviceMemory((void **)&userEuclideanNorm_d, sizeof(float) * ratingMatrix.userEuclideanNorm.size());
    //allocate memory for output
    allocateDeviceMemory((void **)&output_d, sizeof(float) * (dim * dim));   

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

    //allocate memory for similarities in host
    similarityMatrix.similarities = (float *) malloc(sizeof(float) * (dim * dim));

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
  
    ////BASIC KERNEL////
 
    // cudaEventRecord(start);
    // csrSimilarityKernel<<<grid_dim, block_dim>>>(dim, csrRowPtr_d, csrColIdx_d, csrData_d, userEuclideanNorm_d, output_d);
    // cudaEventRecord(stop);

    // //display results of kernel 1
    // copyFromDeviceMemory(similarityMatrix.similarities, output_d, sizeof(float) * (dim * dim));
    // cudaEventSynchronize(stop);

    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Basic Kernel time: %f ms \n",  milliseconds);

    ////COARSENED KERNEL////

    // unsigned int bestThreadCount = ((dim-1)*dim)/2;
    // cudaEventRecord(start);
    // csrSimilarityKernelCoarsened<<<ceil(bestThreadCount/1024.0f), 1024>>>(dim, csrRowPtr_d, csrColIdx_d, csrData_d, userEuclideanNorm_d, output_d);
    // cudaEventRecord(stop);

    // //display results of kernel 1
    // copyFromDeviceMemory(similarityMatrix.similarities, output_d, sizeof(float) * (dim * dim));
    // cudaEventSynchronize(stop);

    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Coarsened Kernel time: %f ms \n",  milliseconds);
    
    ////SHARED KERNEL////

    cudaEventRecord(start);
    csrSimilarityKernelShared<<<dim, BLOCK_SIZE>>>(dim, csrRowPtr_d, csrColIdx_d, csrData_d, userEuclideanNorm_d, output_d);
    cudaEventRecord(stop);

    //display results of kernel 2
    copyFromDeviceMemory(similarityMatrix.similarities, output_d, sizeof(float) * (dim * dim));
    cudaEventSynchronize(stop);


    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Shared Kernel time: %f ms \n", milliseconds);

    //cudaFree
    cudaFree(csrRowPtr_d); 
    cudaFree(csrColIdx_d);
    cudaFree(csrData_d);
    cudaFree(userEuclideanNorm_d);
    cudaFree(output_d);

    return similarityMatrix;
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

