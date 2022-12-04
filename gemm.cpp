#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 1000


/*
    1, 2, 3     1, 2      22, 28
    4, 5, 6     3, 4      49, 64 
                5, 6
*/


void gemm(float* matrix_a, float* matrix_b, float* result, int a_rows, int a_cols, int b_cols)
{
    for (int i = 0; i < a_rows; i++) {
        // printf("\rmul percentage: %.3f%%", 100. * (i + 1) / a_rows);
        for (int j = 0; j < b_cols; j++) {
            for (int k = 0; k < a_cols; k++) {
                result[i * a_rows + j] += matrix_a[i * a_cols + k] * matrix_b[k * b_cols + j];
            }
        }
    }
    printf("\n");
}


void opt_gemm(float* matrix_a, float* matrix_b, float* result, int a_rows, int a_cols, int b_cols)
{
    for (int i = 0; i < a_rows; i++) {
        // printf("\rmul percentage: %.3f%%", 100. * (i + 1) / a_rows);
        for (int k = 0; k < a_cols; k++) {
            for (int j = 0; j < b_cols; j++) {
                result[i * a_rows + j] += matrix_a[i * a_cols + k] * matrix_b[k * b_cols + j];
            }
        }
    }
    printf("\n");
}


void omp_gemm(float* matrix_a, float* matrix_b, float* result, int a_rows, int a_cols, int b_cols)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < a_rows; i++) {
        // printf("\rmul percentage: %.3f%%", 100. * (i + 1) / a_rows);
        for (int k = 0; k < a_cols; k++) {
            for (int j = 0; j < b_cols; j++) {
                result[i * a_rows + j] += matrix_a[i * a_cols + k] * matrix_b[k * b_cols + j];
            }
        }
    }
    // printf("\n");
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


void benchmark(int suite_length)
{
    float** groupa = (float**) malloc(suite_length * sizeof(float*));
    float** groupb = (float**) malloc(suite_length * sizeof(float*));
    float** groupc = (float**) malloc(suite_length * sizeof(float*));

    gen_test_matrix(groupa, suite_length);
    gen_test_matrix(groupb, suite_length);

    for (int i = 0; i < suite_length; i++) {
        float* curr_mem = (float*) malloc(SIZE * SIZE * sizeof(float));
        memset(curr_mem, 0, SIZE * SIZE * sizeof(float));
        groupc[i] = curr_mem;
    }

    printf("test suites generate done, starting gemm...\n", suite_length, SIZE, SIZE);
    float start = clock();
    for (int i = 0; i < suite_length; i++) {
        // printf("\r%d/%d complete,\n", i + 1, suite_length);
        omp_gemm(groupa[i], groupb[i], groupc[i], SIZE, SIZE, SIZE);
    }
    float end = clock();
    float running_secs = (end - start) / CLOCKS_PER_SEC;
    printf("%d %dx%d gemm end, using %f secs.\n", suite_length, SIZE, SIZE, running_secs);
    printf("performance: %.3f GFLOPS.\n", suite_length * 2 / running_secs);
}


int main(int argc, char* argv[])
{
    int test_len = atoi(argv[1]);

    benchmark(test_len);
    return 0;
}