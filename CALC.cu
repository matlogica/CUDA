#include <stdio.h>
#include <math.h>

#ifdef __CUDACC__
// This code is compiled by the CUDA compiler
#ifdef __CUDA_ARCH__
// GPU-specific code: __CUDA_ARCH__ is defined, so we are compiling for the device

typedef double Real;
#else
// CPU-specific code: __CUDA_ARCH__ is not defined, so we are compiling for the host

#include <aadc/idouble.h>
#include <aadc/aadc.h>
typedef idouble Real;

#define HOST_ONLY_CODE
#endif
#endif

#include <iostream>

// Analytics callable from CPU and GPU
__host__ __device__ void analytics(Real a, Real b, Real &c) {
        c = a * std::exp(std::sin(b));
}

__global__ void CUDAKernel(Real *a, Real *b, Real *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        analytics(a[index], b[index], c[index]);
    }
}

int CALC() {
    int n = 256; // Size of the vector
    Real *a, *b, *c; // Pointers for host memory
    Real *d_a, *d_b, *d_c; // Pointers for device memory

    // Allocate host memory
    a = (Real*)malloc(n * sizeof(Real));
    b = (Real*)malloc(n * sizeof(Real));
    c = (Real*)malloc(n * sizeof(Real));

    // Initialize arrays with some values
    for(int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(Real));
    cudaMalloc(&d_b, n * sizeof(Real));
    cudaMalloc(&d_c, n * sizeof(Real));

    // Copy inputs to device
    cudaMemcpy(d_a, a, n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(Real), cudaMemcpyHostToDevice);

    // Launch kernel on default stream without arguments
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    CUDAKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(c, d_c, n * sizeof(Real), cudaMemcpyDeviceToHost);

    // Print results
    for(int i = 0; i < n; i++) {
        std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "CUDA kernel executed successfully" << std::endl;

#ifdef HOST_ONLY_CODE
    std::cout << "Running AADC" << std::endl;

    /// AADC code
    
    auto da = (Real*)malloc(n * sizeof(Real));
    auto db = (Real*)malloc(n * sizeof(Real));


    typedef __m256d mmType;

    aadc::AADCFunctions<mmType> aadc_func;    

    aadc_func.startRecording();
    // Mark 1st elements of a and b as input
    aadc::AADCArgument arg_a(a[0].markAsInput()), arg_b(b[0].markAsInput());
    
    analytics(a[0], b[0], c[0]); // Call analytics function on 1st instance of data

    // Mark 1st element of c as output
    aadc::AADCResult res_c(c[0].markAsOutput());
    aadc_func.stopRecording();

    // Execute AADC kernel with derivatives

    int num_avx_elements = n / aadc::mmSize<mmType>();

    std::shared_ptr<aadc::AADCWorkSpace<mmType> > ws(aadc_func.createWorkSpace());

    for(int i = 0; i < num_avx_elements; i++) {
        for (int avx_i = 0; avx_i < aadc::mmSize<mmType>(); ++avx_i) {
            ws->valp(arg_a)[avx_i] = AAD_PASSIVE(a[i * aadc::mmSize<mmType>() + avx_i]);
            ws->valp(arg_b)[avx_i] = AAD_PASSIVE(b[i * aadc::mmSize<mmType>() + avx_i]);
        }
        aadc_func.forward(*ws);
        for (int avx_i = 0; avx_i < aadc::mmSize<mmType>(); ++avx_i) {
            c[i * aadc::mmSize<mmType>() + avx_i] = ws->valp(res_c)[avx_i];
        }
        ws->setDiff(res_c, 1.0);
        aadc_func.reverse(*ws);

        for (int avx_i = 0; avx_i < aadc::mmSize<mmType>(); ++avx_i) {
            da[i * aadc::mmSize<mmType>() + avx_i] = ws->diffp(arg_a)[avx_i];
            db[i * aadc::mmSize<mmType>() + avx_i] = ws->diffp(arg_b)[avx_i];
        }
    }

    // Print results
    for(int i = 0; i < n; i++) {
        std::cout << a[i] << " " << b[i] << " " << c[i] << " " << da[i] << " " << db[i] << std::endl;
    }

    // Free host memory
    free(da);
    free(db);

#endif

    // Free host memory
    free(a);
    free(b);
    free(c);


    return 0;
}
