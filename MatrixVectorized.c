#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h> // Required for SIMD (AVX) instructions

// Determinist generator
void generate_matrix(double **matrix, int n, long seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// 1. Scalar baseline (optimized with i-k-j for spacial locality)
void multiply_scalar(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) C[i][j] = 0.0;
        for (int k = 0; k < n; k++) {
            double a_val = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += a_val * B[k][j];
            }
        }
    }
}

// 2. Optional extension: Explicit vectorization (AVX2 + FMA)
// Process 4 'double' (256 bits) variables for each clock cycle
void multiply_vectorized(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) C[i][j] = 0.0;
        
        for (int k = 0; k < n; k++) {
            // Load one A value and double 4 times in a 256 bits vector
            __m256d a_vec = _mm256_set1_pd(A[i][k]);
            
            // Advance in blocks of 4 columns (SIMD lane width for double)
            for (int j = 0; j < n; j += 4) {
                // Cargar 4 valores contiguos de B y C
                __m256d b_vec = _mm256_loadu_pd(&B[k][j]);
                __m256d c_vec = _mm256_loadu_pd(&C[i][j]);
                
                // FMA: c_vec = (a_vec * b_vec) + c_vec in a single hardware instruction
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                
                // Save 4 return results in C
                _mm256_storeu_pd(&C[i][j], c_vec);
            }
        }
    }
}

// Check correctness
int check_correctness(int n) {
    double **A = malloc(n * sizeof(double *));
    double **I = malloc(n * sizeof(double *));
    double **C_vec = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        A[i] = malloc(n * sizeof(double));
        I[i] = malloc(n * sizeof(double));
        C_vec[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) I[i][j] = (i == j) ? 1.0 : 0.0;
    }

    generate_matrix(A, n, 42);
    multiply_vectorized(A, I, C_vec, n);

    double epsilon = 1e-9;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(C_vec[i][j] - A[i][j]) > epsilon) return 0;
        }
    }
    return 1;
}

int main() {
    if (!check_correctness(12)) {
        printf("Error: SIMD Vectorization correctness validation failed.\n");
        return 1;
    }
    printf("Validation successful: SIMD Kernel matches identity matrix.\n");

    int n = 500; // Same fix size used at threads analysis
    int repetitions = 5;
    
    FILE *f = fopen("results_vectorized.csv", "w");
    fprintf(f, "method,n,repetition,time_seconds\n");

    // Reservar memoria
    double **A = malloc(n * sizeof(double *));
    double **B = malloc(n * sizeof(double *));
    double **C = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        A[i] = malloc(n * sizeof(double));
        B[i] = malloc(n * sizeof(double));
        C[i] = malloc(n * sizeof(double));
    }

    printf("\n--- Running SIMD Vectorization Benchmark (n = %d) ---\n", n);

    // 1. Measure Scalar
    for (int rep = 1; rep <= repetitions; rep++) {
        generate_matrix(A, n, 100 + rep);
        generate_matrix(B, n, 200 + rep);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        multiply_scalar(A, B, C, n);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        fprintf(f, "Scalar,%d,%d,%.6f\n", n, rep, elapsed);
        printf("Method: Scalar     | Rep: %d | Time: %.4f s\n", rep, elapsed);
    }

    // 2. Measure vectorized
    for (int rep = 1; rep <= repetitions; rep++) {
        generate_matrix(A, n, 300 + rep);
        generate_matrix(B, n, 400 + rep);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        multiply_vectorized(A, B, C, n);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        fprintf(f, "Vectorized,%d,%d,%.6f\n", n, rep, elapsed);
        printf("Method: Vectorized | Rep: %d | Time: %.4f s\n", rep, elapsed);
    }

    fclose(f);
    printf("Benchmark complete. Data saved to results_vectorized.csv\n");
    return 0;
}