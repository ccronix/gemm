#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#define SIZE 1000
#define SUB_MAT 16


/*
    1, 2, 3     1, 2      22, 28
    4, 5, 6     3, 4      49, 64 
                5, 6
*/


__host__
void cpu_gemm(float* matrix_a, float* matrix_b, float* result, int a_rows, int a_cols, int b_cols)
{
    for (int i = 0; i < a_rows; i++) {
        // printf("\rmul percentage: %.3f%%", 100. * (i + 1) / a_rows);
        for (int j = 0; j < b_cols; j++) {
            for (int k = 0; k < a_cols; k++) {
                result[i * a_rows + j] += matrix_a[i * a_cols + k] * matrix_b[k * b_cols + j];
            }
        }
    }
    // printf("\n");
}


__global__
void cuda_gemm_kernel(float* matrix_a, float* matrix_b, float* result, int a_rows, int a_cols, int b_cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= a_rows || j >= b_cols) return;
    
    float sum = 0;
    for (int k = 0; k < a_cols; k++) {
        sum += matrix_a[i * a_cols + k] * matrix_b[k * b_cols + j];
    }
    result[i * a_rows + j] = sum;
}


void printm(float* matrix, int matrix_rows, int matrix_cols)
{
    float curr_val = 0;
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < matrix_cols; j++) {
            curr_val = matrix[i * matrix_cols + j];
            printf("%.3f ", curr_val);
        }
        printf("\n");
    }
    printf("\n");
}


void random_matrix(float* genm)
{
    for (int i = 0; i < SIZE * SIZE; i++) {
        float random_float = (float) rand() / RAND_MAX;
        genm[i] = random_float;
    }
}

void gen_test_matrix(float** test_mat, int array_len)
{
    for (int i = 0; i < array_len; i++){
        float* curr_mat = (float*) malloc(SIZE * SIZE * sizeof(float));
        random_matrix(curr_mat);
        test_mat[i] = curr_mat;
    }
}


__host__
void cuda_benchmark(int suite_length)
{
    printf("alloating host memory...\n");

    float** groupa = (float**) malloc(suite_length * sizeof(float*));
    float** groupb = (float**) malloc(suite_length * sizeof(float*));
    float** groupc = (float**) malloc(suite_length * sizeof(float*));

    printf("host memory done, generating test suite...\n");

    gen_test_matrix(groupa, suite_length);
    gen_test_matrix(groupb, suite_length);

    printf("test suite done, allocating result memory...\n");

    for (int i = 0; i < suite_length; i++) {
        float* curr_mem = (float*) malloc(SIZE * SIZE * sizeof(float));
        groupc[i] = curr_mem;
    }

    printf("result memory done, allocating gpu memory...\n");

    float* gpu_groupa, * gpu_groupb, * gpu_groupc;
    cudaMalloc(& gpu_groupa, SIZE * SIZE * sizeof(float));
    cudaMalloc(& gpu_groupb, SIZE * SIZE * sizeof(float));
    cudaMalloc(& gpu_groupc, SIZE * SIZE * sizeof(float));

    printf("allocate gpu memory done, starting kernel...\n");

    double start = clock();

    for (int i = 0; i < suite_length; i++) {
        cudaMemcpy(gpu_groupa, groupa[i], SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_groupb, groupb[i], SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice);

        const int dim = 8;

        dim3 dimGrid(SIZE / dim, SIZE / dim, 1); 
        dim3 dimBlock(dim, dim, 1);

        cuda_gemm_kernel<<<dimGrid, dimBlock>>>(gpu_groupa, gpu_groupb, gpu_groupc, SIZE, SIZE, SIZE);
        cudaDeviceSynchronize();

        cudaMemcpy(groupc[i], gpu_groupc, SIZE * SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    }

    double end = clock();
    double running_secs = (end - start) / CLOCKS_PER_SEC;

    printf("%d %dx%d gemm end, using %f secs.\n", suite_length, SIZE, SIZE, running_secs);

    printf("performance: %.3f GFLOPS.\n", suite_length * 2 / running_secs);
    printf("kernel done.\n");

    cudaFree(gpu_groupa);
    cudaFree(gpu_groupb);
    cudaFree(gpu_groupc);

}


int main(int argc, char* argv[])
{
    int test_len = atoi(argv[1]);
    cuda_benchmark(test_len);
    return 0;
}