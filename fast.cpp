#include <iostream>
#include <omp.h>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "openacc.h"
#include <random>
#include <cusolverMg.h>
#include <cuda_runtime.h>
#include <vector>
#include "utilities.h"
#include <cublas_v2.h>
#include <stdexcept>
#include <cublasXt.h>

#define ROUND_UP(x) ( x )
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
 
using namespace std;
using namespace std::chrono;
 
const int num_gpus = acc_get_num_devices(acc_device_nvidia);

half BASICALLY_ZERO = .00001;
 
void print_matrix(float *A, int lda, int m, int n){
    for(int i =0; i < m; i++){
        for (int j = 0; j < n; j++)
            cout << setiosflags(ios::fixed) <<setprecision(4) << A[lda*j+i] << setw(6) << "\t";
        cout<<endl;
    }
}

void svd(float *A, int m, int n, float *U, float *V, float *S,int num_gpus){
    /*
    Calculates At*A
    */
    float alpha = 1.0, beta = 0.0;
    cublasXtHandle_t handle_xt;

    vector<int> device_list(num_gpus);
    for (int i = 0; i < num_gpus; ++i)
        device_list[i] = i;
    
    enablePeerAccess(num_gpus,device_list.data());
    
    cublasXtCreate(&handle_xt);
    
    cublasXtDeviceSelect(handle_xt, num_gpus, device_list.data());
    
    cublasXtSgemm(handle_xt, 
                CUBLAS_OP_T, 
                CUBLAS_OP_N, 
                m, 
                n, 
                m, 
                &alpha, 
                A, 
                m, 
                A, 
                m, 
                &beta, 
                V, 
                n);
    
    cudaDeviceSynchronize();

    /*
    Calculates the eigen values and vectors of AtA
    */


    // assume AtA is a float * of length n*n and is filled out

    // Set up device list
    
    // Copy data to GPUs
    
    std::vector<cudaStream_t> streams(num_gpus);
    cusolverMgHandle_t cusolverMgH { nullptr };
    cusolverMgCreate( &cusolverMgH );
    cusolverMgDeviceSelect( cusolverMgH, num_gpus, device_list.data() );
    
    // Create a grid for the matrix distribution
    
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;
    
    cudaLibMgMatrixDesc_t   descrA;
    cudaLibMgGrid_t         gridA;
    cusolverMgCreateDeviceGrid( &gridA, 1, num_gpus, device_list.data(), mapping );
    cusolverMgCreateMatrixDesc( &descrA,
                                              n,   /* number of rows of (global) A */
                                              n,   /* number of columns of (global) A */
                                              n,   /* number or rows in a tile */
                                              256, /* number of columns in a tile */
                                              CUDA_R_32F,
                                              gridA ) ;
    vector<float *>   array_d_A( num_gpus,nullptr );
    CreateMat( num_gpus,
               device_list.data(),
               n,   /* number of columns of global A */
               256, /* number of columns per column tile */
               n, /* leading dimension of local A */
               array_d_A.data() );

    MemcpyH2D<float>( num_gpus,
               device_list.data(),
               n,
               n, /* input */
               V,
               n,               /* output */
               n,                 /* number of columns of global A */
               256,               /* number of columns per column tile */
               n,               /* leading dimension of local A */
               array_d_A.data(), /* host pointer array of dimension num_devices */
               1,
               1 );


    vector<float> W(n, 0.0);

    // Allocate workspace
    cusolverStatus_t status;
    int64_t lwork = 0;
    status = cusolverMgSyevd_bufferSize(cusolverMgH, 
                                CUSOLVER_EIG_MODE_VECTOR, 
                                CUBLAS_FILL_MODE_LOWER, 
                                n, 
                                reinterpret_cast<void **>( array_d_A.data( ) ), 
                                1,
                                1, 
                                descrA, 
                                reinterpret_cast<void *>(W.data()), 
                                CUDA_R_32F,
                                CUDA_R_32F, 
                                &lwork);
    
    if (status != CUSOLVER_STATUS_SUCCESS) {
        // Handle the error
        printf("cusolverMgSyevd_bufferSize failed with status %d\n", status);
    } 
    

    // Perform eigen decomposition
    
    std::vector<float *> array_d_work( num_gpus,nullptr );

    /* array_d_work[j] points to device workspace of device j */
    WorkspaceAlloc( num_gpus,
                    device_list.data(),
                    sizeof( float ) * lwork, /* number of bytes per device */
                    reinterpret_cast<void **>(array_d_work.data()));
    cudaDeviceSynchronize();

    int info;
    cusolverMgSyevd(cusolverMgH, 
                    CUSOLVER_EIG_MODE_VECTOR, 
                    CUBLAS_FILL_MODE_LOWER, 
                    n, 
                    reinterpret_cast<void **>( array_d_A.data( ) ), 
                    1,
                    1, 
                    descrA, 
                    reinterpret_cast<void **>(W.data()),
                    CUDA_R_32F, 
                    CUDA_R_32F,
                    reinterpret_cast<void **>( array_d_work.data( ) ), 
                    lwork,
                    &info);
    cudaDeviceSynchronize();

    if (info != 0) {
        throw std::runtime_error("Eigen decomposition failed.");
    }

    MemcpyD2H<float>(num_gpus, device_list.data(), n,n,
                         /* input */
                         n,   /* number of columns of global A */
                         256, /* number of columns per column tile */
                         n, /* leading dimension of local A */
                         array_d_A.data(), 1, 1,
                         /* output */
                         V, /* N-y-N eigenvectors */
                         n);
    
    cusolverMgDestroyMatrixDesc(descrA);
    cusolverMgDestroyGrid(gridA);
    cusolverMgDestroy(cusolverMgH);
    for (long int i = 0; i < num_gpus; ++i) {
        cudaFree(array_d_A[i]);
        cudaFree(array_d_work[i]);
        cudaStreamDestroy(streams[i]);
    }

    float *scaled_V = new float[n*n];
    
    #pragma omp parallel for
    for( long int i = 0 ; i < MIN(n,m); i++){
        if (W[i] > .0000001)
            S[i] = sqrt(W[i]);
        else
            S[i] = 0.0;
    }

    //cout<<endl;
    // #pragma omp parallel for
    for(long int i = 0 ; i < n*n; i++){
        if (S[i%n] > .0000001)
            scaled_V[i] =V[i]/S[ (i/ n) ] ;
        else
            scaled_V[i] = 0.0;
    }

    /*
    
     compute U
    
    */

    // Initialize cublas handles and device memory
    
    cublasXtSgemm(handle_xt, 
                CUBLAS_OP_N, 
                CUBLAS_OP_N, 
                m, 
                n, 
                n, 
                &alpha, 
                A, 
                m, 
                scaled_V, 
                n, 
                &beta, 
                U, 
                m);
    
    cudaDeviceSynchronize();
    cublasXtDestroy(handle_xt);
    
    delete[] scaled_V;
}
 
 
int main(int argc, char* argv[]){
    acc_init(acc_device_nvidia);
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1, 1);
 
    long int m = stoi(argv[1]), n = stoi(argv[2]),gpus = stoi(argv[3]);
   
    float *A = new  float[m*n];
    float *V = new float[m*n];
    float *S = new float[MIN(n,m)];
    
    #pragma omp parallel for
    for(long int i =0; i < n; i++){
        for (long int j = 0; j < m; j++)
            A[m*i+j] = distribution(generator);
    }


    auto start = high_resolution_clock::now();
    
    svd(A,m,n,A,V,S,gpus);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " seconds" << endl;
    
    return 0;
}
