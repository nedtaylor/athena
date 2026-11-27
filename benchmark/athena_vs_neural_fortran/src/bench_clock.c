#include <time.h>

double bench_monotonic(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}
