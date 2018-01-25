#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cublas_v2.h>


#define SEED 1337
#define MATRIX_DIM 200

using namespace std;

float randInRange(float min, float max)
{
  return min + (float) (rand() / (double) (RAND_MAX + 1) * (max - min + 1));
}

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void fill_matrix(float *A, int nr_rows_A, int nr_cols_A, float min, float max) {
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            A[j * nr_rows_A + i] = randInRange(min, max);
        }
    }
}

bool check_matrices(float *A, float *B, int dim){
    for(int i = 0; i < dim; ++i){
        for(int j = 0; j < dim; ++j){
            if (abs(A[j * dim + i] - B[j * dim + i]) > 0.01) {
                cout << i << "," << j << endl;
                cout << A[j * dim + i] << "," << B[j * dim + i] << endl;
                return false;
            } 
        }
    }
    return true;
}

int main() {
    // Allocate 3 arrays on CPU
    srand(SEED);
    
    int dim = MATRIX_DIM;

    float *A = (float *)malloc(dim * dim * sizeof(float));
    float *B = (float *)malloc(dim * dim * sizeof(float));
    float *C = (float *)malloc(dim * dim * sizeof(float));

    float *c_C = (float *)malloc(dim * dim * sizeof(float));

    fill_matrix(A, dim, dim, 1.0, 10.0);
    fill_matrix(B, dim, dim, 1.0, 10.0);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time; 
    cudaEventRecord(start, 0); 

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,dim * dim * sizeof(float));
    cudaMalloc(&d_B,dim * dim * sizeof(float));
    cudaMalloc(&d_C,dim * dim * sizeof(float));

    cudaMemcpy(d_A,A,dim * dim * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,dim * dim * sizeof(float),cudaMemcpyHostToDevice);


    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, d_A, dim, d_B, dim, beta, d_C, dim);

    cublasDestroy(handle);

    cudaMemcpy(C, d_C, dim * dim * sizeof(float),cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "GPU time is: " << time/1000 << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);  


    clock_t Time;
    Time = clock(); 


    int c, d, k;
    float sum = 0;
    for (c = 0; c < dim; c++) {
      for (d = 0; d < dim; d++) {
        for (k = 0; k < dim; k++) {
          sum = sum + A[k*dim+c] * B[d * dim + k];
        }
        //cout << sum << endl;
        c_C[d*dim+c] = sum;
        sum = 0;
      }
    }

    Time = clock() - Time;
    float Time_ = (float) Time / CLOCKS_PER_SEC;
    cout << "CPU time is: " <<  Time_ << endl; 


    cout << check_matrices(c_C, C, dim) << endl;

    cout << true << endl;


    free(A);
    free(B);
    free(C);
    free(c_C);

    return 0;
}