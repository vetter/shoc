#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cudacommon.h"
#include <cassert>
#include <iostream>
#include <vector>
#include "OptionParser.h"
#include "ResultDatabase.h"

#include <cublas.h>

#define TRAINING_SIZE 5000
#define TEST_SIZE 1000

#define IMAGE_SIZE 784

float ONE=1.0;
float ZERO=0.0;

float *BIASES; 
float *WEIGHTS;

float *D__BIASES; 
float *D__WEIGHTS;

float *NABLA_B; 
float *NABLA_W;

float *D__NABLA_B; 
float *D__NABLA_W;

float *DELTA_NABLA_B; 
float *DELTA_NABLA_W;

float *D__DELTA_NABLA_B; 
float *D__DELTA_NABLA_W;

int MINI_BATCH_SIZE;

int TBS;
int TWS;

float ETA;

int NUM_LAYERS;
int *SIZES;

float **ACTIVATIONS;
float **ZS;

float *ACTIVATIONS_2D;

float **D__ACTIVATIONS_2D_TRAINING;
float **D__ZS_2D_TRAINING;

float **D__ACTIVATIONS_2D_TEST;
float **D__ZS_2D_TEST;

float **TRAINING_DATA_X;
int *TRAINING_DATA_Y;

float *D__TRAINING_DATA_X_2D;

float **TEST_DATA_X;
int *TEST_DATA_Y;

float *D__TEST_DATA_X_2D;
float **D__TEST_DATA_X;

int I,J;

__global__ void kernelUpdateBiases(float *nabla_b,float *biases,float eta,float mini_batch_size) {

  float rate=eta/mini_batch_size;

  biases[threadIdx.x]-=rate*nabla_b[threadIdx.x];
}

__global__ void kernelUpdateWeights(float *nabla_w,float *weights,int tws,float eta,float mini_batch_size) {

  float rate=eta/mini_batch_size;

  if ((blockIdx.x*blockDim.x+threadIdx.x)<tws) {
    weights[blockIdx.x*blockDim.x+threadIdx.x]-=rate*nabla_w[blockIdx.x*blockDim.x+threadIdx.x];
  }
}

__global__ void kernelUpdateNablaB(float *nabla_b,float *delta_nabla_b) {
  nabla_b[threadIdx.x]+=delta_nabla_b[threadIdx.x];
}

__global__ void kernelUpdateNablaW(float *nabla_w,float *delta_nabla_w,int tws) {
  if ((blockIdx.x*blockDim.x+threadIdx.x)<tws) {
    nabla_w[blockIdx.x*blockDim.x+threadIdx.x]+=delta_nabla_w[blockIdx.x*blockDim.x+threadIdx.x];
  }
}

__global__ void kernelInitNablaB(float *nabla_b) {
  nabla_b[threadIdx.x]=0.0;
}

__global__ void kernelInitNablaW(float *nabla_w,int tws) {
  if ((blockIdx.x*blockDim.x+threadIdx.x)<tws) {
    nabla_w[blockIdx.x*blockDim.x+threadIdx.x]=0.0;
  }
}

__global__ void kernelBackprop3a(float *delta_nabla_b,int b_off,int bound,int b_off_old,float *weights,int w_off_old) {

  int j;

  delta_nabla_b[b_off+threadIdx.x]=0.0;
  for (j=0; j<bound; j++) {
    delta_nabla_b[b_off+threadIdx.x]+=delta_nabla_b[b_off_old+j]*weights[w_off_old+(j*blockDim.x)+threadIdx.x];
  }
}

__global__ void kernelBackprop3b(float *delta_nabla_b,int b_off,float *zs) {
  delta_nabla_b[b_off+threadIdx.x]*=(1.0/(1.0+expf(-zs[threadIdx.x])))*(1.0-(1.0/(1.0+expf(-zs[threadIdx.x]))));
}

__global__ void kernelBackprop1(float *delta_nabla_w,int w_off,float *activations,float *delta_nabla_b,int b_off) {
  delta_nabla_w[w_off+(blockIdx.x*blockDim.x)+threadIdx.x]=activations[threadIdx.x]*delta_nabla_b[b_off+blockIdx.x];
  //delta_nabla_w[w_off+(threadIdx.x*gridDim.x)+blockIdx.x]=activations[threadIdx.x]*delta_nabla_b[b_off+blockIdx.x];
}

__global__ void kernelBackprop2(float *delta_nabla_b,int b_off,float *activations,float *zs,float y) {

  int y_[10]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

  y_[(int)(y+0.1)]=1.0;
  delta_nabla_b[b_off+threadIdx.x]=(activations[threadIdx.x]-y_[threadIdx.x])*(1.0/(1.0+expf(-zs[threadIdx.x])))*(1.0-(1.0/(1.0+expf(-zs[threadIdx.x]))));
}

__global__ void kernelFeedForward1(float *zs,int bound2,float *weights,int w_off,float *activations1) {

  int i;

  zs[threadIdx.x]=0.0;
  for (i=0; i<bound2; i++) {
    zs[threadIdx.x]+=weights[w_off+(threadIdx.x*bound2)+i]*activations1[i];
  }
}

__global__ void kernelFeedForward1b(float *zs,int bound,float *weights,int w_off,float *activations) {

  int i;

  zs[(blockIdx.x*blockDim.x)+threadIdx.x]=0.0;
  for (i=0; i<bound; i++) {
    zs[(blockIdx.x*blockDim.x)+threadIdx.x]+=weights[w_off+(threadIdx.x*bound)+i]*activations[(blockIdx.x*bound)+i];
  }
}

__global__ void kernelFeedForward3(float *zs,float *biases,int b_off,float *activations) {
  zs[(blockIdx.x*blockDim.x)+threadIdx.x]+=biases[b_off+threadIdx.x];
  activations[(blockIdx.x*blockDim.x)+threadIdx.x]=1.0/(1.0+expf(-zs[(blockIdx.x*blockDim.x)+threadIdx.x]));
}

__global__ void kernelFeedForward2(float *zs,float *biases,int b_off,float *activations) {
  zs[threadIdx.x]+=biases[b_off+threadIdx.x];
  activations[threadIdx.x]=1.0/(1.0+expf(-zs[threadIdx.x]));
}

void cublasCheck(cublasStatus_t status, const char *fn_name)
{
  if(status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr,"cublas error returned %d from %s. exiting...\n", status, fn_name);
    exit(EXIT_FAILURE);
  }
}

float normal() {

  int i;
  float uniform=0.0;

  for (i=0; i<12; i++) {
    uniform+=(float)rand()/(float)RAND_MAX;  
  }
  return uniform-6.0;

}

float sigmoid(float z) {

  return 1.0/(1.0+expf(-z));

}

float sigmoid_prime(float z) {

  return sigmoid(z)*(1.0-sigmoid(z));

}

void backprop(float y) {

  int i,k;
  int b_offset=0,w_offset=0,w_offset_old,b_offset_old;

  if (I==0) {

    D__ACTIVATIONS_2D_TRAINING[0]=&D__TRAINING_DATA_X_2D[J*MINI_BATCH_SIZE*IMAGE_SIZE];

    // Feed forward using all training sets in mini batch
    for (k=1; k<NUM_LAYERS; k++) {

      cublasSgemm('t','n',SIZES[k],MINI_BATCH_SIZE,SIZES[k-1],1,&D__WEIGHTS[w_offset],SIZES[k-1],D__ACTIVATIONS_2D_TRAINING[k-1],SIZES[k-1],0,D__ZS_2D_TRAINING[k-1],SIZES[k]);

      {dim3 dimBlock(SIZES[k],1,1); dim3 dimGrid(MINI_BATCH_SIZE,1,1);
      kernelFeedForward3<<< dimGrid, dimBlock >>>(D__ZS_2D_TRAINING[k-1],D__BIASES,b_offset,D__ACTIVATIONS_2D_TRAINING[k]);}

      b_offset+=SIZES[k];
      w_offset+=SIZES[k-1]*SIZES[k];
    }

  }

  b_offset=0;
  w_offset=0;
  for (i=1; i<(NUM_LAYERS-1); i++) {
    b_offset+=SIZES[i];
    w_offset+=SIZES[i-1]*SIZES[i];
  }

  {dim3 dimBlock(SIZES[NUM_LAYERS-1],1,1); dim3 dimGrid(1,1,1);
  kernelBackprop2<<< dimGrid, dimBlock >>>(D__DELTA_NABLA_B,b_offset,&D__ACTIVATIONS_2D_TRAINING[NUM_LAYERS-1][I*SIZES[NUM_LAYERS-1]],
                                           &D__ZS_2D_TRAINING[NUM_LAYERS-2][I*SIZES[NUM_LAYERS-1]],y);}

  {dim3 dimBlock(SIZES[NUM_LAYERS-2],1,1); dim3 dimGrid(SIZES[NUM_LAYERS-1],1,1);
  kernelBackprop1<<< dimGrid, dimBlock >>>(D__DELTA_NABLA_W,w_offset,&D__ACTIVATIONS_2D_TRAINING[NUM_LAYERS-2][I*SIZES[NUM_LAYERS-2]],D__DELTA_NABLA_B,b_offset);}

  for (k=2; k<NUM_LAYERS; k++) {

    b_offset_old=b_offset;
    b_offset-=SIZES[NUM_LAYERS-k];
    w_offset_old=w_offset;
    w_offset-=SIZES[NUM_LAYERS-k]*SIZES[NUM_LAYERS-k-1];

    {dim3 dimBlock(SIZES[NUM_LAYERS-k],1,1); dim3 dimGrid(1,1,1);
    kernelBackprop3a<<< dimGrid, dimBlock >>>(D__DELTA_NABLA_B,b_offset,SIZES[NUM_LAYERS-k+1],b_offset_old,D__WEIGHTS,w_offset_old);}

    {dim3 dimBlock(SIZES[NUM_LAYERS-k],1,1); dim3 dimGrid(1,1,1);
    kernelBackprop3b<<< dimGrid, dimBlock >>>(D__DELTA_NABLA_B,b_offset,&D__ZS_2D_TRAINING[NUM_LAYERS-k-1][I*SIZES[NUM_LAYERS-k]]);}

    {dim3 dimBlock(SIZES[NUM_LAYERS-k-1],1,1); dim3 dimGrid(SIZES[NUM_LAYERS-k],1,1);
    kernelBackprop1<<< dimGrid, dimBlock >>>(D__DELTA_NABLA_W,w_offset,&D__ACTIVATIONS_2D_TRAINING[NUM_LAYERS-k-1][I*SIZES[NUM_LAYERS-k-1]],D__DELTA_NABLA_B,b_offset);}

  }

}

void update_mini_batch(int *mini_batch_y) {

  int i;

  {dim3 dimBlock(TBS,1,1); dim3 dimGrid(1,1,1);
  kernelInitNablaB<<< dimGrid, dimBlock >>>(D__NABLA_B);}

  {dim3 dimBlock(1024,1,1); dim3 dimGrid((TWS/1024)+1,1,1);
  kernelInitNablaW<<< dimGrid, dimBlock >>>(D__NABLA_W,TWS);}

  for (i=0; i<MINI_BATCH_SIZE; i++) {

    I=i;

    backprop(mini_batch_y[i]);

    cublasSaxpy(TBS,1,D__DELTA_NABLA_B,1,D__NABLA_B,1);

    cublasSaxpy(TWS,1,D__DELTA_NABLA_W,1,D__NABLA_W,1);

  }

  cublasSaxpy(TBS,-(ETA/MINI_BATCH_SIZE),D__NABLA_B,1,D__BIASES,1);

  cublasSaxpy(TWS,-(ETA/MINI_BATCH_SIZE),D__NABLA_W,1,D__WEIGHTS,1);

}

void shuffle(float **d__x, float **x, int *y, int n) {

  int i,swap,itmp;
  float *ftmp;

  for (i=n-1; i>0; i--) {

    swap=rand()/(RAND_MAX/(i+1)+1);

    ftmp=d__x[swap];
    d__x[swap]=d__x[i];
    d__x[i]=ftmp;

    ftmp=x[swap];
    x[swap]=x[i];
    x[i]=ftmp;

    itmp=y[swap];
    y[swap]=y[i];
    y[i]=itmp;

  }
}

#ifdef __DEVICE_EMULATION__
#define __SYNC __syncthreads();
#else
#define __SYNC ;
#endif

using namespace std;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    ;
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the neural net benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing, results are stored in resultDB
//
// Programmer: Mitch Horton
// Creation: December 1st, 2014
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{

  long i,j;
  NUM_LAYERS=3;
  SIZES=(int*)malloc(NUM_LAYERS*sizeof(int));
  SIZES[0]=784;
  SIZES[1]=30;
  SIZES[2]=10;
  int num_epochs=10;
  MINI_BATCH_SIZE=10;
  ETA=3.0;

  float percent_correct;

  //printf("%d %d %f\n",num_epochs,MINI_BATCH_SIZE,ETA);


  FILE *f_training_data_x = fopen("nn_data/training_data_x","r");
  FILE *f_training_data_y = fopen("nn_data/training_data_y","r");
  FILE *f_test_data_x = fopen("nn_data/test_data_x","r");
  FILE *f_test_data_y = fopen("nn_data/test_data_y","r");

  if(f_training_data_x == NULL)
  {
    cout<<"Input training file not found - please check data directory!"<<endl;
    return;
  }

  TRAINING_DATA_X=(float**)malloc(TRAINING_SIZE*sizeof(float*));
  for (i=0; i<TRAINING_SIZE; i++) {
    TRAINING_DATA_X[i] = (float*)malloc(IMAGE_SIZE*sizeof(float));
  }

  CUDA_SAFE_CALL(cudaMalloc((void**)&D__TRAINING_DATA_X_2D,TRAINING_SIZE*IMAGE_SIZE*sizeof(float)));

  TEST_DATA_X=(float**)malloc(10000*sizeof(float*));
  for (i=0; i<TEST_SIZE; i++) {
    TEST_DATA_X[i] = (float*)malloc(IMAGE_SIZE*sizeof(float));
  }

  CUDA_SAFE_CALL(cudaMalloc((void**)&D__TEST_DATA_X_2D,TEST_SIZE*IMAGE_SIZE*sizeof(float)));

  D__TEST_DATA_X=(float**)malloc(TEST_SIZE*sizeof(float*));
  for (i=0; i<TEST_SIZE; i++) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&D__TEST_DATA_X[i],IMAGE_SIZE*sizeof(float)));
  }

  TRAINING_DATA_Y = (int*)malloc(TRAINING_SIZE*sizeof(int));
  TEST_DATA_Y = (int*)malloc(TEST_SIZE*sizeof(int));

  ACTIVATIONS=(float**)malloc(NUM_LAYERS*sizeof(float*));
  ZS =(float**)malloc((NUM_LAYERS-1)*sizeof(float*));

  D__ACTIVATIONS_2D_TRAINING=(float**)malloc(NUM_LAYERS*sizeof(float*));
  D__ZS_2D_TRAINING =(float**)malloc((NUM_LAYERS-1)*sizeof(float*));

  D__ACTIVATIONS_2D_TEST=(float**)malloc(NUM_LAYERS*sizeof(float*));
  D__ZS_2D_TEST =(float**)malloc((NUM_LAYERS-1)*sizeof(float*));

  for (i=1; i<NUM_LAYERS; i++) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&D__ACTIVATIONS_2D_TRAINING[i],SIZES[i]*MINI_BATCH_SIZE*sizeof(float)));
  }

  for (i=1; i<NUM_LAYERS; i++) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&D__ACTIVATIONS_2D_TEST[i],SIZES[i]*TEST_SIZE*sizeof(float)));
  }

  ACTIVATIONS_2D=(float*)malloc(SIZES[NUM_LAYERS-1]*TEST_SIZE*sizeof(float));

  for (i=1; i<NUM_LAYERS; i++) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&D__ZS_2D_TRAINING[i-1],SIZES[i]*MINI_BATCH_SIZE*sizeof(float)));
  }

  for (i=1; i<NUM_LAYERS; i++) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&D__ZS_2D_TEST[i-1],SIZES[i]*TEST_SIZE*sizeof(float)));
  }

  for (i=1; i<NUM_LAYERS; i++) {
    ACTIVATIONS[i]=(float*)malloc(SIZES[i]*10*sizeof(float));
  }

  for (i=1; i<NUM_LAYERS; i++) {
    ZS[i-1]=(float*)malloc(SIZES[i]*10*sizeof(float));
  }

  for (i=0; i<TRAINING_SIZE; i++) {
    fread(TRAINING_DATA_X[i],sizeof(float),IMAGE_SIZE,f_training_data_x);
  }
  fclose(f_training_data_x);

  for (i=0; i<TEST_SIZE; i++) {
    fread(TEST_DATA_X[i],sizeof(float),IMAGE_SIZE,f_test_data_x);
  }
  fclose(f_test_data_x);

  fread(TRAINING_DATA_Y,sizeof(int),TRAINING_SIZE,f_training_data_y);
  fclose(f_training_data_y);

  fread(TEST_DATA_Y,sizeof(int),TEST_SIZE,f_test_data_y);
  fclose(f_test_data_y);

  TBS=0;
  TWS=0;

  for (i=1; i<NUM_LAYERS; i++) {
    TBS+=SIZES[i];
    TWS+=SIZES[i-1]*SIZES[i];
  }

  BIASES =(float*)malloc(TBS*sizeof(float));
  WEIGHTS=(float*)malloc(TWS*sizeof(float));

  CUDA_SAFE_CALL(cudaMalloc((void**)&D__BIASES,TBS*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&D__WEIGHTS,TWS*sizeof(float)));

  srand(7777);

  for (i=0; i<TBS; i++) {
    BIASES[i]=normal();
  }
  for (i=0; i<TWS; i++) {
    WEIGHTS[i]=normal();
  }

  NABLA_B =(float*)malloc(TBS*sizeof(float));
  NABLA_W=(float*)malloc(TWS*sizeof(float));

  DELTA_NABLA_B =(float*)malloc(TBS*sizeof(float));
  DELTA_NABLA_W=(float*)malloc(TWS*sizeof(float));

  CUDA_SAFE_CALL(cudaMalloc((void**)&D__NABLA_B,TBS*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&D__NABLA_W,TWS*sizeof(float)));

  CUDA_SAFE_CALL(cudaMalloc((void**)&D__DELTA_NABLA_B,TBS*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&D__DELTA_NABLA_W,TWS*sizeof(float)));

  cublasInit();

  ////int probSizes[4] = { 1, 8, 48, 96 };
  ////int size = probSizes[op.getOptionInt("size")-1];

  int iterations = op.getOptionInt("passes");

  iterations=2;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int it = 0; it < iterations; it++) {

    // Copy inputs to GPU
    double transferTime = 0.;
    cudaEventRecord(start, 0);

    for (i=0; i<TRAINING_SIZE; i++) {
      CUDA_SAFE_CALL(cudaMemcpy(&D__TRAINING_DATA_X_2D[i*IMAGE_SIZE], TRAINING_DATA_X[i], sizeof(float)*IMAGE_SIZE, cudaMemcpyHostToDevice));
    }

    for (i=0; i<TEST_SIZE; i++) {
      CUDA_SAFE_CALL(cudaMemcpy(&D__TEST_DATA_X_2D[i*IMAGE_SIZE], TEST_DATA_X[i], sizeof(float)*IMAGE_SIZE, cudaMemcpyHostToDevice));
    }

    for (i=0; i<TEST_SIZE; i++) {
      CUDA_SAFE_CALL(cudaMemcpy(D__TEST_DATA_X[i], TEST_DATA_X[i], sizeof(float)*IMAGE_SIZE, cudaMemcpyHostToDevice));
    }

    CUDA_SAFE_CALL(cudaMemcpy(TEST_DATA_X[1], &D__TEST_DATA_X_2D[1*IMAGE_SIZE], sizeof(float)*IMAGE_SIZE, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(D__BIASES, BIASES, sizeof(float)*TBS, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(D__WEIGHTS, WEIGHTS, sizeof(float)*TWS, cudaMemcpyHostToDevice));

    cudaEventRecord(stop, 0);
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    cudaEventRecord(start, 0);

    // loop over epochs
    for (i=0; i<num_epochs; i++) {

      //shuffle(D__TRAINING_DATA_X,TRAINING_DATA_X,TRAINING_DATA_Y,TRAINING_SIZE);

      // loop over mini-batches
      for (j=0; j<TRAINING_SIZE/MINI_BATCH_SIZE; j++) {

        J=j;

        update_mini_batch(&TRAINING_DATA_Y[j*MINI_BATCH_SIZE]);

      }

///*

      if (i == (num_epochs-1)) {

        // test current weights and biases
        int total_correct=0,highest;
        long k,b_offset,w_offset;

        D__ACTIVATIONS_2D_TEST[0]=D__TEST_DATA_X_2D;

        b_offset=w_offset=0;

        for (k=1; k<NUM_LAYERS; k++) {

          cublasSgemm('t','n',SIZES[k],TEST_SIZE,SIZES[k-1],1,&D__WEIGHTS[w_offset],SIZES[k-1],D__ACTIVATIONS_2D_TEST[k-1],SIZES[k-1],0,D__ZS_2D_TEST[k-1],SIZES[k]);

          {dim3 dimBlock(SIZES[k],1,1); dim3 dimGrid(TEST_SIZE,1,1);
          kernelFeedForward3<<< dimGrid, dimBlock >>>(D__ZS_2D_TEST[k-1],D__BIASES,b_offset,D__ACTIVATIONS_2D_TEST[k]);}

          b_offset+=SIZES[k];
          w_offset+=SIZES[k-1]*SIZES[k];
        }

        CUDA_SAFE_CALL(cudaMemcpy(ACTIVATIONS_2D, D__ACTIVATIONS_2D_TEST[NUM_LAYERS-1], sizeof(float)*SIZES[NUM_LAYERS-1]*TEST_SIZE, cudaMemcpyDeviceToHost));

        for (j=0; j<TEST_SIZE; j++) {

          highest=0;
          for (k=1; k<SIZES[NUM_LAYERS-1]; k++) {
            if (ACTIVATIONS_2D[(j*SIZES[NUM_LAYERS-1])+k]>ACTIVATIONS_2D[(j*SIZES[NUM_LAYERS-1])+highest]) {
              highest=k;
            }
          }

          if (highest==TEST_DATA_Y[j]) {
            total_correct++;
          }

        }

        printf("%d %d %d\n",(int)i,total_correct,TEST_SIZE);

        percent_correct=(float)(total_correct)/(float)(TEST_SIZE);

      }

//*/

    }

    cudaEventRecord(stop, 0);
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&elapsedTime, start, stop);
    double kernelTime = elapsedTime * 1.e-3;

    // Test to make sure neural net reached threshold; if not, return
    if (percent_correct<.8) {
      return;
    }

    char atts[1024];
    sprintf(atts, "%d training_sets", TRAINING_SIZE*num_epochs);
    resultDB.AddResult("Learning-Rate", atts, "training_sets/s", TRAINING_SIZE*num_epochs / kernelTime);
    resultDB.AddResult("Learning-Rate_PCIe", atts, "training_sets/s", TRAINING_SIZE*num_epochs / (transferTime+kernelTime));
    }
    
    // Clean up


    free(BIASES);
    free(WEIGHTS);
    free(NABLA_B);
    free(NABLA_W);
    free(DELTA_NABLA_B);
    free(DELTA_NABLA_W);
    free(ACTIVATIONS_2D);
    free(SIZES);
    free(TEST_DATA_Y);
    free(TRAINING_DATA_Y);

    for (i=0; i<NUM_LAYERS; i++) {
      free(ACTIVATIONS[i]);
    }
    free(ACTIVATIONS);

    for (i=0; i<NUM_LAYERS; i++) {
      free(ZS[i]);
    }
    free(ZS);

    for (i=1; i<NUM_LAYERS; i++) {
      CUDA_SAFE_CALL(cudaFree(D__ACTIVATIONS_2D_TRAINING[i]));
    }
    free(D__ACTIVATIONS_2D_TRAINING);

    for (i=1; i<NUM_LAYERS; i++) {
      CUDA_SAFE_CALL(cudaFree(D__ZS_2D_TEST[i]));
    }
    free(D__ZS_2D_TEST);

    for (i=1; i<NUM_LAYERS; i++) {
      CUDA_SAFE_CALL(cudaFree(D__ZS_2D_TRAINING[i]));
    }
    free(D__ZS_2D_TRAINING);

    for (i=0; i<TEST_SIZE; i++) {
      CUDA_SAFE_CALL(cudaFree(D__TEST_DATA_X[i]));
    }
    free(D__TEST_DATA_X);

    for (i=1; i<NUM_LAYERS; i++) {
      CUDA_SAFE_CALL(cudaFree(D__ACTIVATIONS_2D_TEST[i]));
    }
    free(D__ACTIVATIONS_2D_TEST);

    for (i=0; i<TRAINING_SIZE; i++) {
      free(TRAINING_DATA_X[i]);
    }
    free(TRAINING_DATA_X);

    for (i=0; i<TEST_SIZE; i++) {
      free(TEST_DATA_X[i]);
    }
    free(TEST_DATA_X);

    CUDA_SAFE_CALL(cudaFree(D__BIASES));
    CUDA_SAFE_CALL(cudaFree(D__WEIGHTS));
    CUDA_SAFE_CALL(cudaFree(D__NABLA_B));
    CUDA_SAFE_CALL(cudaFree(D__NABLA_W));
    CUDA_SAFE_CALL(cudaFree(D__DELTA_NABLA_B));
    CUDA_SAFE_CALL(cudaFree(D__DELTA_NABLA_W));
    CUDA_SAFE_CALL(cudaFree(D__TRAINING_DATA_X_2D));
    CUDA_SAFE_CALL(cudaFree(D__TEST_DATA_X_2D));

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

}

