// mpi_summa.cpp
// Scalable Universal Matrix Multiplication Algorithm (SUMMA)
// 2D block decomposition: processes arranged in a sqrt(p) x sqrt(p) grid
// Each process holds a block of A and a block of B, computes block of C
// Communication: broadcasts along process rows/columns (reduces total communication)
//
// Assumes N is divisible by block_dim (process grid dimension)
// For simplicity, uses sqrt(p) x sqrt(p) process grid

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

// Simple matrix multiply: C += A*B (all are m x k x n blocks)
void local_matmul(int m, int k, int n, const double* A, const double* B, double* C) {
    for(int i=0;i<m;++i) {
        for(int j=0;j<n;++j) {
            for(int p=0;p<k;++p) {
                C[i*n+j] += A[i*k+p] * B[p*n+j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s N [num_blocks]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    int N = atoi(argv[1]);
    int num_blocks = (argc > 2) ? atoi(argv[2]) : (int)sqrt((double)size);
    
    if (num_blocks * num_blocks != size) {
        if (rank == 0) fprintf(stderr, "Error: num_blocks^2 (%d) must equal comm size (%d)\n", num_blocks * num_blocks, size);
        MPI_Finalize();
        return 1;
    }
    
    if (N % num_blocks != 0) {
        if (rank == 0) fprintf(stderr, "Error: N (%d) must be divisible by num_blocks (%d)\n", N, num_blocks);
        MPI_Finalize();
        return 1;
    }
    
    int block_size = N / num_blocks;
    int my_row = rank / num_blocks;
    int my_col = rank % num_blocks;
    
    // Create row and column communicators for broadcasts
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);
    
    // Allocate local blocks
    std::vector<double> A_block(block_size * block_size, 0.0);
    std::vector<double> B_block(block_size * block_size, 0.0);
    std::vector<double> C_block(block_size * block_size, 0.0);
    
    // Full matrices only on rank 0
    std::vector<double> fullA, fullB, fullC;
    if (rank == 0) {
        fullA.resize((size_t)N*N);
        fullB.resize((size_t)N*N);
        fullC.resize((size_t)N*N, 0.0);
        
        srand48(12345);
        for (int i=0; i<N*N; ++i) {
            fullA[i] = drand48();
            fullB[i] = drand48();
        }
    }
    
    double start_time = MPI_Wtime();
    
    // Distribute blocks: rank 0 sends A and B blocks to all ranks
    if (rank == 0) {
        // copy own block
        int a_row_start = (my_row * block_size) * N;
        int a_col_start = my_col * block_size;
        for (int i=0; i<block_size; ++i) {
            for (int j=0; j<block_size; ++j) {
                A_block[i*block_size+j] = fullA[a_row_start + i*N + a_col_start + j];
            }
        }
        
        int b_row_start = (my_row * block_size) * N;
        int b_col_start = my_col * block_size;
        for (int i=0; i<block_size; ++i) {
            for (int j=0; j<block_size; ++j) {
                B_block[i*block_size+j] = fullB[b_row_start + i*N + b_col_start + j];
            }
        }
        
        // send blocks to other ranks
        for (int r=1; r<size; ++r) {
            int r_row = r / num_blocks;
            int r_col = r % num_blocks;
            
            // send A block
            a_row_start = (r_row * block_size) * N;
            a_col_start = r_col * block_size;
            std::vector<double> tmp_a(block_size * block_size);
            for (int i=0; i<block_size; ++i) {
                for (int j=0; j<block_size; ++j) {
                    tmp_a[i*block_size+j] = fullA[a_row_start + i*N + a_col_start + j];
                }
            }
            MPI_Send(tmp_a.data(), block_size*block_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            
            // send B block
            b_row_start = (r_row * block_size) * N;
            b_col_start = r_col * block_size;
            std::vector<double> tmp_b(block_size * block_size);
            for (int i=0; i<block_size; ++i) {
                for (int j=0; j<block_size; ++j) {
                    tmp_b[i*block_size+j] = fullB[b_row_start + i*N + b_col_start + j];
                }
            }
            MPI_Send(tmp_b.data(), block_size*block_size, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(A_block.data(), block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_block.data(), block_size*block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // SUMMA: for each panel k in [0, num_blocks)
    //   broadcast A[my_row][k] along row
    //   broadcast B[k][my_col] along column
    //   compute local C += A_bcast * B_bcast
    for (int k=0; k<num_blocks; ++k) {
        std::vector<double> A_bcast(block_size * block_size, 0.0);
        std::vector<double> B_bcast(block_size * block_size, 0.0);
        
        // Process in row my_row sends its A block (column k) along the row
        if (my_col == k) {
            A_bcast = A_block;
        }
        MPI_Bcast(A_bcast.data(), block_size*block_size, MPI_DOUBLE, k, row_comm);
        
        // Process in column my_col sends its B block (row k) along the column
        if (my_row == k) {
            B_bcast = B_block;
        }
        MPI_Bcast(B_bcast.data(), block_size*block_size, MPI_DOUBLE, k, col_comm);
        
        // Local multiplication: C += A_bcast * B_bcast
        local_matmul(block_size, block_size, block_size,
                     A_bcast.data(), B_bcast.data(), C_block.data());
    }
    
    // Gather all C blocks to rank 0
    if (rank == 0) {
        // copy own block
        int c_row_start = (my_row * block_size) * N;
        int c_col_start = my_col * block_size;
        for (int i=0; i<block_size; ++i) {
            for (int j=0; j<block_size; ++j) {
                fullC[c_row_start + i*N + c_col_start + j] = C_block[i*block_size+j];
            }
        }
        
        // receive C blocks from other ranks
        for (int r=1; r<size; ++r) {
            std::vector<double> tmp_c(block_size * block_size);
            MPI_Recv(tmp_c.data(), block_size*block_size, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            int r_row = r / num_blocks;
            int r_col = r % num_blocks;
            c_row_start = (r_row * block_size) * N;
            c_col_start = r_col * block_size;
            for (int i=0; i<block_size; ++i) {
                for (int j=0; j<block_size; ++j) {
                    fullC[c_row_start + i*N + c_col_start + j] = tmp_c[i*block_size+j];
                }
            }
        }
    } else {
        MPI_Send(C_block.data(), block_size*block_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
    
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    
    // Compute checksum on rank 0
    if (rank == 0) {
        double csum = 0.0;
        for (int i=0; i<N*N; ++i) csum += fullC[i];
        printf("n=%d processes=%d time(s)=%.6f checksum=%.6e\n", N, size, elapsed, csum);
    }
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
