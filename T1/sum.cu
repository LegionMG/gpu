#include <stdlib.h>
#include <iostream>
#include <time.h>


using namespace std;

#define BLOCK_SIZE 256 
#define SEED 1337
#define ARR_LENGTH 5000

__global__ void sum_cud(float * in, float * out, int len) {
    __shared__ float sum[2*BLOCK_SIZE];
    unsigned int th_num = threadIdx.x;
    unsigned int pointer = blockIdx.x * blockDim.x;
    if (pointer + th_num < len)
       sum[th_num] = in[pointer + th_num];
    else
       sum[th_num] = 0;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride >>= 1) {
       if (th_num < stride)
          sum[th_num] += sum[th_num+stride];
       __syncthreads();
    }
    if (th_num == 0)
       out[blockIdx.x] = sum[0];
}



float randInRange(float min, float max)
{
  return min + (float) (rand() / (double) (RAND_MAX + 1) * (max - min + 1));
}

int main(int argc, char ** argv) {
    srand(SEED);
    float * input;
    float * output;
    float * d_input;
    float * d_output;
    int lenInput = ARR_LENGTH; 
    int lenOutput; 

    input = (float*) malloc(lenInput * sizeof(float));

    for (int i = 0; i < lenInput; ++i)
    {
        input[i] = randInRange(0.0, 10.0);
    }

    clock_t Time;
    Time = clock(); 

    float sum = 0.0;
    for (int i = 0; i < lenInput; i++) {
        sum += input[i];
    }

    cout << "CPU result: " << sum << endl;

    Time = clock() - Time;
    float Time_ = (float) Time / CLOCKS_PER_SEC;
    cout << "CPU time is: " <<  Time_ << endl; 




    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_; 
    cudaEventRecord(start, 0);  
    do
    {   

        lenOutput = lenInput / (BLOCK_SIZE);
        if (lenInput % (BLOCK_SIZE)) {
            lenOutput++;
        }
        output = (float*) malloc(lenOutput * sizeof(float));

        cudaMalloc(&d_input, sizeof(float) * lenInput);
        cudaMalloc(&d_output, sizeof(float) * lenInput);


        cudaMemcpy(d_input, input, sizeof(float) * lenInput, cudaMemcpyHostToDevice);

        dim3 dimGrid(lenOutput);
        dim3 dimBlock(BLOCK_SIZE);


        sum_cud<<<dimGrid, dimBlock>>>(d_input, d_output, lenInput);

        cudaMemcpy(output, d_output, sizeof(float) * lenOutput, cudaMemcpyDeviceToHost);
        input = output;
        lenInput = lenOutput; 

    } while (lenOutput > 1);
   
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_, start, stop);
    printf("GPU ime is: %f\n", time_/1000); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    cout << "Cuda result: " << output[0] << endl;

    cudaFree(d_input);
    cudaFree(d_output);


    free(input);
    free(output);


    return 0;
}