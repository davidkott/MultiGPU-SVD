#include <iostream>
#include <omp.h>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "openacc.h"
#include <random>
#include <cuda.h>
#include <cublas_v2.h>

#define ROUND_UP(x) ( x + 32 - (x%32) )
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
 
using namespace std;
using namespace std::chrono;
 
/*
 
  Using column domanint format of matrices since I had to pick something
 
  Using unified CUDA memory, if you don't then you're fucked
 
*/
 
const int num_gpus = acc_get_num_devices(acc_device_nvidia);

float BASICALLY_ZERO = .00001;
 
void print_matrix(float *A, int lda, int m, int n){
    for(int i =0; i < m; i++){
        for (int j = 0; j < n; j++)
            cout << setiosflags(ios::fixed) <<setprecision(4) << A[lda*j+i]<< setw(6) << "\t";
        cout<<endl;
    }
}

void svd(float *A, int m, int n, int lda, float *U, float *V, float *S){
    int ldaTa = ROUND_UP(n);

    float *AtA = new float[ldaTa*n];
    float *Q = new float[ldaTa*n];
    float *R = new float[ldaTa*n];
    float *temp = new float[ldaTa*n];
    float *V_hat = new float[ldaTa*n];
    float tollerance = 1e-4;
    int max_iter = 1000;
    float sum;

    #pragma acc data copyin(A[0:lda*m], lda, ldaTa, n, m) copyout(AtA[0:ldaTa*n])
    {
    #pragma acc parallel loop gang
    for (int matmul_j = 0; matmul_j < n; matmul_j++) {
            #pragma acc loop worker
            for (int matmul_i = 0; matmul_i < n; matmul_i++){
                sum = 0;
                #pragma acc loop vector reduction(+:sum)
                for (int matmul_k = 0; matmul_k < m; matmul_k++)
                    sum += A[matmul_k+ lda * matmul_i] * A[matmul_k + lda * matmul_j];
                if (abs(sum) < BASICALLY_ZERO)
                    sum = 0.0;
                AtA[matmul_i + ldaTa * matmul_j] = sum;
            }
        }
    }
    
    for(int mM = 0; mM < n; mM++){
        for(int nN = 0; nN < n; nN++)
            S[nN*ldaTa+mM] = AtA[nN*ldaTa+mM];
    }
    
    for (int eigen_decomp_i = 0; eigen_decomp_i < max_iter; eigen_decomp_i++) {

        /*
        TODO: find a better way to handle this
        */
        
        #pragma omp parallel for
        for(int nN = 0; nN < n; nN++){
            for(int mM = 0; mM < n; mM++)
                temp[nN*ldaTa+mM] = S[nN*ldaTa+mM];
        }



        #pragma acc data copyin(temp[0:ldaTa*n],ldaTa,R[0:ldaTa*n], Q[0:ldaTa*n]) copyout(S[0:ldaTa*n])
        {
            for (int qr_k = 0; qr_k < n; qr_k++ ) {

                float product = 0;
                #pragma acc loop vector reduction(+:product)
                for (int qr_i = 0; qr_i < n; qr_i++)
                    product += temp[ldaTa*qr_k + qr_i] * temp[ldaTa*qr_k + qr_i];

                R[ldaTa*qr_k+qr_k] = sqrt(product);
                if (abs(R[ldaTa*qr_k+qr_k]) < BASICALLY_ZERO){
                    #pragma acc loop vector
                    for (int qr_i = 0; qr_i < n; qr_i++)
                        Q[ldaTa*qr_k +qr_i] = 0;
                    R[ldaTa*qr_k+qr_k] = 0.0;
                }
                else{
                    #pragma acc loop vector
                    for (int qr_i = 0; qr_i < n; qr_i++)
                        Q[ldaTa*qr_k + qr_i] = temp[ldaTa*qr_k + qr_i]/R[ldaTa*qr_k+qr_k];
                }   
                
                #pragma acc loop independent
                for (int qr_j = qr_k+1; qr_j < n; qr_j++) {
                    float product = 0;
                    #pragma acc loop vector reduction(+:product)
                    for (int qr_i = 0; qr_i < n; qr_i++)
                        product += Q[ldaTa*qr_k+qr_i] * temp[ldaTa*qr_j + qr_i];
                    R[ldaTa*qr_j+qr_k] = product;
                    #pragma acc loop vector
                    for (int qr_i = 0; qr_i < n; qr_i++)
                        temp[ldaTa*qr_j + qr_i] = temp[ldaTa*qr_j + qr_i] - R[ldaTa*qr_j+qr_k] * Q[ldaTa*qr_k + qr_i];
                }
            }
            #pragma acc parallel loop  gang worker collapse(2)
            for (int matmul_j = 0; matmul_j < n; matmul_j++) {
                for (int matmul_i = 0; matmul_i < n; matmul_i++){
                    sum = 0;
                    #pragma acc loop vector reduction(+:sum)
                    for (int matmul_k = 0; matmul_k < n; matmul_k++)
                        sum += R[matmul_i + ldaTa * matmul_k] * Q[matmul_k + ldaTa * matmul_j];
                    if (abs(sum) < BASICALLY_ZERO)
                        sum = 0.0;
                    S[matmul_i + ldaTa * matmul_j] = sum;
                }
            }
        }
        
        if(eigen_decomp_i == 0){
            #pragma omp parallel for
            for(int nN = 0; nN < n; nN++){
                for(int mM = 0; mM < n; mM++)
                    V[nN*ldaTa+mM] = Q[nN*ldaTa+mM];
            }
        } else{
            #pragma acc parallel loop gang
            for (int matmul_j = 0; matmul_j < n; matmul_j++) {
                #pragma acc loop worker
                for (int matmul_i = 0; matmul_i < n; matmul_i++){
                    sum = 0;
                    #pragma acc loop vector reduction(+:sum)
                    for (int matmul_k = 0; matmul_k < n; matmul_k++)
                        sum += V[matmul_i + ldaTa * matmul_k] * Q[matmul_k + ldaTa * matmul_j];
                    if (abs(sum) < BASICALLY_ZERO)
                        sum = 0.0;
                    V_hat[matmul_i + ldaTa * matmul_j] = sum;
                }
            }
            float* temp2 = V;
            V = V_hat;
            V_hat = temp2;
            
        }
        float product = 0.0;
        #pragma acc data copyin(S[0:ldaTa*m], ldaTa, n, m) copy(product)
        {
            // Adjust the number of gangs and the vector length as needed
            #pragma acc parallel loop gang vector_length(256) reduction(+:product)
            for (int diag_norm_i = 0; diag_norm_i < m; diag_norm_i++){
                for(int diag_norm_j = 0; diag_norm_j < n; diag_norm_j++)
                    product += S[ldaTa*diag_norm_j+diag_norm_i] *  S[ldaTa*diag_norm_j+diag_norm_i];
            }
            #pragma acc parallel loop gang vector_length(256) reduction(+:product)
            for (int diag_norm_i = 0; diag_norm_i < m; diag_norm_i++)
                product -= S[ldaTa*diag_norm_i+diag_norm_i] *  S[ldaTa*diag_norm_i+diag_norm_i];
        }

    
        if (product <= tollerance ){
            break;
        }
 
    }

    for (int i = 0; i < n; i++){
        if (S[lda*i+i] > BASICALLY_ZERO)
            S[lda*i+i] = sqrt(S[lda*i+i]);
        else
            S[lda*i+i] = 0;
    }
    for (int j = 0; j < m; j++) {
            for (int i = 0; i < m; i++){
                float sum = 0;
                // #pragma acc loop vector reduction(+:sum)
                for (int k = 0; k < n; k++)
                    sum += A[i + lda * k] * V[k + ldaTa * j];
                if (abs(sum) < BASICALLY_ZERO)
                    sum = 0.0;
                U[i + lda * j] = sum;
            }
        }

    for (int mat_scale_j = 0; mat_scale_j < n; mat_scale_j++) {
        for (int mat_scale_i = 0; mat_scale_i < m; mat_scale_i++){
            if (S[mat_scale_j + lda * mat_scale_j] > BASICALLY_ZERO)
                U[mat_scale_j*lda + mat_scale_i] *= 1/S[mat_scale_j + lda * mat_scale_j];
            else
                U[mat_scale_j*lda + mat_scale_i] = 0;
        }
    }

    print_matrix(S,lda,5,5);
    

    delete[] AtA;
    delete []Q;
    delete []R;
    delete []V_hat;
    delete []temp;
}
 
 
int main(int argc, char* argv[]){
    acc_init(acc_device_nvidia);
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1, 1);
 
    int m = stoi(argv[1]), n = stoi(argv[2]);
    int lda = ROUND_UP(m);
   
     float *A = new float[lda*n];
    float *U = new float[lda*n];
    float *V = new float[lda*n];
    float *S = new float[lda*n];
 
 
    for(int i =0; i < m; i++){
        for (int j = 0; j < n; j++)
            A[lda*j+i] = distribution(generator);
    }
    
    auto start = high_resolution_clock::now();
    
    svd(A,m,n,lda,U,V,S);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " seconds" << endl;
    
    delete[] A;
    delete[] U;
    delete[] V;
    delete[] S;
    return 0;
}
