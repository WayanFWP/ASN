#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define M_PI 3.14159265358979323846

typedef struct {
    double real;
    double imag;
} Complex;

#ifdef __cplusplus
extern "C" {
#endif

double complex* fft(const double complex* x, int N) {
    if (N <= 1) {
        double complex* out = malloc(sizeof(double complex));
        out[0] = x[0];
        return out;
    }

    double complex* even = malloc(N/2 * sizeof(double complex));
    double complex* odd  = malloc(N/2 * sizeof(double complex));
    for (int i = 0; i < N/2; i++) {
        even[i] = x[2*i];
        odd[i]  = x[2*i + 1];
    }

    double complex* Feven = fft(even, N/2);
    double complex* Fodd  = fft(odd, N/2);
    free(even); free(odd);

    double complex* X = malloc(N * sizeof(double complex));
    for (int k = 0; k < N/2; k++) {
        double complex t = cexp(-2.0 * I * M_PI * k / N) * Fodd[k];
        X[k]       = Feven[k] + t;
        X[k+N/2]   = Feven[k] - t;
    }

    free(Feven); free(Fodd);
    return X;
}

void free_memory(void* ptr) {
    free(ptr);
}

#ifdef __cplusplus
}
#endif
