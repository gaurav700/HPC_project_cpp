#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

void fillMatrix(vector<double> &A, int N) {
    for (int i = 0; i < N * N; ++i) A[i] = (i % 100) * 0.01;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 500;
    int rowsPerProc = N / size;
    int rem = N % size;

    // Each process's local matrix dimensions
    int localRows = rowsPerProc + (rank < rem ? 1 : 0);
    vector<double> A(localRows * N);
    vector<double> B(N * N);
    vector<double> C(localRows * N, 0.0);

    double start = MPI_Wtime();

    // Root initializes full matrices
    if (rank == 0) {
        vector<double> fullA(N * N);
        fillMatrix(fullA, N);
        fillMatrix(B, N);

        // Distribute blocks of A
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int rows = rowsPerProc + (i < rem ? 1 : 0);
            if (i == 0)
                copy(fullA.begin(), fullA.begin() + rows * N, A.begin());
            else
                MPI_Send(fullA.data() + offset * N, rows * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            offset += rows;
        }
    } else {
        MPI_Recv(A.data(), localRows * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Broadcast B to all
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Local multiplication
    for (int i = 0; i < localRows; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i * N + j] += A[i * N + k] * B[k * N + j];

    // Gather results
    vector<int> recvCounts(size), displs(size);
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            recvCounts[i] = (rowsPerProc + (i < rem ? 1 : 0)) * N;
            displs[i] = offset * N;
            offset += rowsPerProc + (i < rem ? 1 : 0);
        }
    }

    vector<double> fullC;
    if (rank == 0) fullC.resize(N * N);
    MPI_Gatherv(C.data(), localRows * N, MPI_DOUBLE,
                fullC.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    // Only root prints timing
    if (rank == 0) {
        double checksum = 0;
        for (double v : fullC) checksum += v;
        cout << "n=" << N << " processes=" << size
             << " time(s)=" << (end - start)
             << " checksum=" << checksum << endl;
    }

    MPI_Finalize();
    return 0;
}
