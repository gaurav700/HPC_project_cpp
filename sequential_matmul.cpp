// sequential_matmul.cpp
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv) {
    int n = 500;
    if (argc > 1) n = stoi(argv[1]);
    vector<double> A(n*n), B(n*n), C(n*n, 0.0);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0,1.0);
    for (int i=0;i<n*n;i++){ A[i]=dist(rng); B[i]=dist(rng);}
    auto t0 = chrono::high_resolution_clock::now();
    for (int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            double s = 0.0;
            for (int k=0;k<n;k++) s += A[i*n + k] * B[k*n + j];
            C[i*n + j] = s;
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(t1-t0).count();
    long double checksum = 0.0L;
    for (double v : C) checksum += v;
    cout << "n="<<n<<" time(s)="<<secs<<" checksum="<< (double)checksum << endl;
    return 0;
}
