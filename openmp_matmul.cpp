// openmp_matmul.cpp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
int main(int argc, char** argv) {
    int n = 500;
    int threads = 4;
    if (argc > 1) n = stoi(argv[1]);
    if (argc > 2) threads = stoi(argv[2]);
    vector<double> A(n*n), B(n*n), C(n*n, 0.0);
    mt19937_64 rng(42);
    uniform_real_distribution<double> dist(0.0,1.0);
    for (int i=0;i<n*n;i++){ A[i]=dist(rng); B[i]=dist(rng);}
    omp_set_num_threads(threads);
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            double s = 0.0;
            for (int k=0;k<n;k++) s += A[i*n + k] * B[k*n + j];
            C[i*n + j] = s;
        }
    }
    double t1 = omp_get_wtime();
    double secs = t1-t0;
    long double checksum = 0.0L;
    for (double v : C) checksum += v;
    cout << "n="<<n<<" threads="<<threads<<" time(s)="<<secs<<" checksum="<< (double)checksum << endl;
    return 0;
}
