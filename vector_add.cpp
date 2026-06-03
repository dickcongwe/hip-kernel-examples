#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK(cmd) { hipError_t e = cmd; if(e != hipSuccess) { printf("HIP error: %s\n", hipGetErrorString(e)); exit(1); } }

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    
    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes), *h_c = (float*)malloc(bytes);
    for(int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = i*2; }
    
    float *d_a, *d_b, *d_c;
    CHECK(hipMalloc(&d_a, bytes)); CHECK(hipMalloc(&d_b, bytes)); CHECK(hipMalloc(&d_c, bytes));
    CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
    
    int threads = 256, blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threads), 0, 0, d_a, d_b, d_c, N);
    CHECK(hipDeviceSynchronize());
    
    CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));
    
    bool ok = true;
    for(int i = 0; i < N; i++) if(h_c[i] != h_a[i] + h_b[i]) { ok = false; break; }
    printf("Vector add %s: %d elements\n", ok ? "PASSED" : "FAILED", N);
    
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
