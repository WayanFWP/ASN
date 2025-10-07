#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

double* convolution(const double *signal, int signal_len,
                    const double *filter, int filter_len) {
    int conv_len = signal_len + filter_len - 1;
    double *conv_result = (double*)malloc(conv_len * sizeof(double));
    
    if (!conv_result) return NULL;

    for (int n = 0; n < conv_len; n++) {
        conv_result[n] = 0.0;
        for (int k = 0; k < filter_len; k++) {
            int idx = n - k;
            if (idx >= 0 && idx < signal_len) {
                conv_result[n] += signal[idx] * filter[k];  // Changed from filter[filter_len - 1 - k]
            }
        }
    }
    return conv_result;
}

#ifdef __cplusplus
}
#endif