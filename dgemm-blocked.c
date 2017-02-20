#include <mm_malloc.h>
#include <string.h>
#include <stdio.h>
const char* dgemm_desc = "Simple blocked dgemm.";


//used for define block size
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define ALIGN_SIZE 64
/* Default for enabling features */
// #define ALIGN_FLOOR 100
// #define PAD_FLOOR 100
#define NAIVE_CEILING 128

/* Default for disabling features */
#define ALIGN_FLOOR 99999
#define PAD_FLOOR 99999
// #define NAIVE_CEILING 0


#define min(a,b) (((a)<(b))?(a):(b))

void block_multiply (int lda, double *A, double *B, double *restrict C);
void naive_multiply (int n, double *A, double *B, double *restrict C);
double *align_matrix (double *A, int size) __attribute__((assume_aligned (ALIGN_SIZE)));
double *pad_matrix (double *A, int old_size, int size_mult);
void print_matrix (double *A, int size);

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* restrict C)
{
  /* For each row i of A */
  for (int j = 0; j < N; ++j)
  {
    /* For each column j of B */
    for (int k = 0; k < K; ++k)
    {
      /* Compute C(i,j) */
      for( int i = 0; i < M; i++ )
      {
        C[i+j*lda] += A[i+k*lda] * B[k+j*lda];;
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double* A, double* B, double* restrict C)
{
  int bytes_in_old = lda * sizeof(double);
  int requires_pad = bytes_in_old % BLOCK_SIZE;
  // Already a multiple of block size, no padding needed
  // if (lda >= PAD_FLOOR && requires_pad != 0)
  // {
  //   int padded_lda = (((lda  * sizeof(double)) / BLOCK_SIZE + 1) * BLOCK_SIZE) / sizeof(double);

  //   double *padded_A = pad_matrix(A, lda, BLOCK_SIZE);

  //   double *padded_B = pad_matrix(B, lda, BLOCK_SIZE);
  //   double *padded_C = pad_matrix(C, lda, BLOCK_SIZE);
  //   block_multiply (padded_lda, padded_A, padded_B, padded_C);
  //   for (int i = 0; i < lda; ++i)
  //   {
  //     memcpy(&C[i * lda], &padded_C[i * padded_lda], sizeof(double) * lda);
  //   }
  //   free(padded_A);
  //   free(padded_B);
  //   free(padded_C);
  // }
  if (lda >= ALIGN_FLOOR)
  {
    double *aligned_A = align_matrix(A, lda);
    double *aligned_B = align_matrix(B, lda);
    double *aligned_C = align_matrix(C, lda);
    C = align_matrix(C, lda);
    __assume_aligned(aligned_A, ALIGN_SIZE);
    __assume_aligned(aligned_B, ALIGN_SIZE);
    __assume_aligned(aligned_C, ALIGN_SIZE);
    block_multiply(lda, aligned_A, aligned_B, aligned_C);

    for (int i = 0; i < lda; ++i)
    {
      memcpy(&C[i * lda], &aligned_C[i * lda], sizeof(double) * lda);
    }

    free(aligned_A);
    free(aligned_B);
    free(aligned_C);
  }
  else if (lda <= NAIVE_CEILING)
  {
    naive_multiply (lda, A, B, C);
  }
  else
  {
    block_multiply (lda, A, B, C);
  }
}

void block_multiply (int lda, double *A, double *B, double *restrict C)
{
  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}

void naive_multiply (int n, double *A, double *B, double *restrict C)
{
  /* For each row i of A */
  for (int j = 0; j < n; ++j)
  {
    /* For each column j of B */
    for (int k = 0; k < n; ++k)
    {
      /* Compute C(i,j) */
      for( int i = 0; i < n; i++ )
      {
        C[i+j*n] += A[i+k*n] * B[k+j*n];;
      }
    }
  }
}

double *align_matrix (double *A, int size)
{
  int bytes = size * size * sizeof(double);
  double *aligned_A = (double *) _mm_malloc(bytes, ALIGN_SIZE);
  memcpy(aligned_A, A, bytes);
  return aligned_A;
}

/* Pad a matrix to a multiple of size_mult. */
double *pad_matrix (double *A, int old_dim_size, int size_mult)
{
  int bytes_in_old = old_dim_size * sizeof(double);
  if (bytes_in_old % size_mult == 0)
  {
    return A;
  }

  int mults_in_old = bytes_in_old / size_mult;
  int new_dim_bytes = (mults_in_old + 1) * size_mult; // New # of bytes per dim
  int new_dim_size = new_dim_bytes / sizeof(double);
  int new_total_bytes = new_dim_bytes * new_dim_bytes / sizeof(double);

  double *padded_A = (double *) calloc(1, new_total_bytes);

  // Copy col by col, since we have extra rows.
  for (int j = 0; j < old_dim_size; ++j) {
    int old_col_index = j * old_dim_size;
    int new_col_index = j * new_dim_size;
    memcpy(&padded_A[new_col_index], &A[old_col_index], sizeof(double) * old_dim_size);
  }
  return padded_A;
}

/* Print a matrix (for debugging) */
void print_matrix (double *A, int size)
{
  // i rows, j cols
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      printf("% 07.4f ", A[i + j * size]);
    }
    printf("\n");
  }
}
