// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Jeffrey Vetter <vetter@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011-2013, UT-Battelle, LLC
// Copyright (c) 2013, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Oak Ridge National Laboratory, nor UT-Battelle, LLC,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "offload.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

//#include <omp.h>

#define TRAINING_SIZE 5000
#define VALIDATION_SIZE 10000
#define TEST_SIZE 1000

#define IMAGE_SIZE 784

__declspec(target(MIC)) int NUM_EPOCHS;

__declspec(target(MIC)) int I;
__declspec(target(MIC)) int J;

__declspec(target(MIC)) float PERCENT_CORRECT;

__declspec(target(MIC)) float *BIASES; 
__declspec(target(MIC)) float *WEIGHTS;

__declspec(target(MIC)) float *NABLA_B; 
__declspec(target(MIC)) float *NABLA_W;

__declspec(target(MIC)) float *DELTA_NABLA_B; 
__declspec(target(MIC)) float *DELTA_NABLA_W;

__declspec(target(MIC)) int MINI_BATCH_SIZE;

__declspec(target(MIC)) int TBS;
__declspec(target(MIC)) int TWS;

__declspec(target(MIC)) float ETA;

__declspec(target(MIC)) int NUM_LAYERS;
__declspec(target(MIC)) int *SIZES;

__declspec(target(MIC)) float **ACTIVATIONS_2D_TRAINING;
__declspec(target(MIC)) float **ZS_2D_TRAINING;

__declspec(target(MIC)) float **ACTIVATIONS_2D_TEST;
__declspec(target(MIC)) float **ZS_2D_TEST;

__declspec(target(MIC)) float *TRAINING_DATA_X_2D;
__declspec(target(MIC)) int *TRAINING_DATA_Y;

__declspec(target(MIC)) float *TEST_DATA_X_2D;
__declspec(target(MIC)) int *TEST_DATA_Y;

__declspec(target(MIC)) float normal() {

  int i;
  float uniform=0.0;

  for (i=0; i<12; i++) {
    uniform+=(float)rand()/(float)RAND_MAX;  
  }
  return uniform-6.0;

}

__declspec(target(MIC)) float sigmoid(float z) {

  return 1.0/(1.0+expf(-z));

}

__declspec(target(MIC)) float sigmoid_prime(float z) {

  return sigmoid(z)*(1.0-sigmoid(z));

}

__declspec(target(MIC)) void backprop(float y) {

  int i,j,k,l;
  int b_offset=0,w_offset=0,w_offset_old,b_offset_old;

  int y_[10]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

  y_[(int)(y+0.1)]=1.0;

  if (I==0) {

    ACTIVATIONS_2D_TRAINING[0]=&TRAINING_DATA_X_2D[J*MINI_BATCH_SIZE*IMAGE_SIZE];

    for (i=0; i<TBS; i++) {
      DELTA_NABLA_B[i]=0.0;
    }

    for (i=0; i<TWS; i++) {
      DELTA_NABLA_W[i]=0.0;
    }

    float c0,c1,c2,c3;

    // Feed forward using given training set
    for (k=1; k<NUM_LAYERS; k++) {

      for (j=0; j<SIZES[k]; j++) {
        for (l=0; l<MINI_BATCH_SIZE; l++) {
          ZS_2D_TRAINING[k-1][l*SIZES[k]+j]=0.0;
        }
      }

#pragma omp parallel for private(l,i,c0,c1,c2,c3)
      for (j=0; j<SIZES[k]; j++) {
        for (l=0; l<MINI_BATCH_SIZE; l++) {
          for (i=0; i<SIZES[k-1]; i=i+4) {
            c0=WEIGHTS[w_offset+(j*SIZES[k-1])+i]*ACTIVATIONS_2D_TRAINING[k-1][l*SIZES[k-1]+i];
            c1                                 =WEIGHTS[w_offset+(j*SIZES[k-1])+i+1]*ACTIVATIONS_2D_TRAINING[k-1][l*SIZES[k-1]+i+1];
            c2                                 =WEIGHTS[w_offset+(j*SIZES[k-1])+i+2]*ACTIVATIONS_2D_TRAINING[k-1][l*SIZES[k-1]+i+2];
            c3                                 =WEIGHTS[w_offset+(j*SIZES[k-1])+i+3]*ACTIVATIONS_2D_TRAINING[k-1][l*SIZES[k-1]+i+3];
            if (k==2 && i==28) {
              ZS_2D_TRAINING[k-1][l*SIZES[k]+j]+=c0+c1;
            } else {
              ZS_2D_TRAINING[k-1][l*SIZES[k]+j]+=c0+c1+c2+c3;
            }
          }
        }
      }

      for (j=0; j<SIZES[k]; j++) {
        for (l=0; l<MINI_BATCH_SIZE; l++) {
          ZS_2D_TRAINING[k-1][l*SIZES[k]+j]+=BIASES[b_offset+j];
          ACTIVATIONS_2D_TRAINING[k][l*SIZES[k]+j]=sigmoid(ZS_2D_TRAINING[k-1][l*SIZES[k]+j]);
        }
      }
      b_offset+=SIZES[k];
      w_offset+=SIZES[k-1]*SIZES[k];
    }

  }

  // back propagate where the error comes from
  b_offset=0;
  w_offset=0;
  for (i=1; i<(NUM_LAYERS-1); i++) {
    b_offset+=SIZES[i];
    w_offset+=SIZES[i-1]*SIZES[i];
  }

  for (i=0; i<SIZES[NUM_LAYERS-1]; i++) {
    DELTA_NABLA_B[b_offset+i]=(ACTIVATIONS_2D_TRAINING[NUM_LAYERS-1][I*SIZES[NUM_LAYERS-1]+i]-y_[i])*sigmoid_prime(ZS_2D_TRAINING[NUM_LAYERS-2][I*SIZES[NUM_LAYERS-1]+i]);
  }

  for (i=0; i<SIZES[NUM_LAYERS-1]; i++) {
    for (j=0; j<SIZES[NUM_LAYERS-2]; j++) {
      DELTA_NABLA_W[w_offset+(i*SIZES[NUM_LAYERS-2])+j]=ACTIVATIONS_2D_TRAINING[NUM_LAYERS-2][I*SIZES[NUM_LAYERS-2]+j]*DELTA_NABLA_B[b_offset+i];
    }
  }

  for (k=2; k<NUM_LAYERS; k++) {

    b_offset_old=b_offset;
    b_offset-=SIZES[NUM_LAYERS-k];
    w_offset_old=w_offset;
    w_offset-=SIZES[NUM_LAYERS-k]*SIZES[NUM_LAYERS-k-1];

    for (i=0; i<SIZES[NUM_LAYERS-k]; i++) {
      DELTA_NABLA_B[b_offset+i]=0.0;
      for (j=0; j<SIZES[NUM_LAYERS-k+1]; j++) {
        DELTA_NABLA_B[b_offset+i]+=DELTA_NABLA_B[b_offset_old+j]*WEIGHTS[w_offset_old+(j*SIZES[NUM_LAYERS-k])+i];
      }
      DELTA_NABLA_B[b_offset+i]*=sigmoid_prime(ZS_2D_TRAINING[NUM_LAYERS-k-1][I*SIZES[NUM_LAYERS-k]+i]);
    }    

#pragma omp parallel for private(j)
    for (i=0; i<SIZES[NUM_LAYERS-k-1]; i++) {
      for (j=0; j<SIZES[NUM_LAYERS-k]; j++) {
        DELTA_NABLA_W[w_offset+(j*SIZES[NUM_LAYERS-k-1])+i]=DELTA_NABLA_B[b_offset+j]*ACTIVATIONS_2D_TRAINING[NUM_LAYERS-k-1][I*SIZES[NUM_LAYERS-k-1]+i];
      }
    }

  }

}

__declspec(target(MIC)) void update_mini_batch(int *mini_batch_y) {

  int i,j;

  for (i=0; i<TBS; i++) {
    NABLA_B[i]=0.0;
  }

  for (i=0; i<TWS; i++) {
    NABLA_W[i]=0.0;
  }

  for (i=0; i<MINI_BATCH_SIZE; i++) {

    I=i;

    backprop(mini_batch_y[i]);

    for (j=0; j<TBS; j++) {
      NABLA_B[j]+=DELTA_NABLA_B[j];
    }

    for (j=0; j<TWS; j++) {
      NABLA_W[j]+=DELTA_NABLA_W[j];
    }

  }


  for (i=0; i<TBS; i++) {
    BIASES[i]-=(ETA/MINI_BATCH_SIZE)*NABLA_B[i];
  }

  for (i=0; i<TWS; i++) {
    WEIGHTS[i]-=(ETA/MINI_BATCH_SIZE)*NABLA_W[i];
  }

}

void shuffle(float **x, int *y, int n) {

  int i,swap,itmp;
  float *ftmp;

  for (i=n-1; i>0; i--) {

    swap=rand()/(RAND_MAX/(i+1)+1);

    ftmp=x[swap];
    x[swap]=x[i];
    x[i]=ftmp;

    itmp=y[swap];
    y[swap]=y[i];
    y[i]=itmp;

  }
}

__declspec(target(MIC)) void main_loop() {

  int i,j,k,l,b_offset,w_offset;

  ACTIVATIONS_2D_TRAINING=(float**)malloc(NUM_LAYERS*sizeof(float*));
  for (i=1; i<NUM_LAYERS; i++) {
    ACTIVATIONS_2D_TRAINING[i]=(float*)malloc(SIZES[i]*MINI_BATCH_SIZE*sizeof(float));
  }

  ZS_2D_TRAINING =(float**)malloc((NUM_LAYERS-1)*sizeof(float*));
  for (i=1; i<NUM_LAYERS; i++) {
    ZS_2D_TRAINING[i-1]=(float*)malloc(SIZES[i]*MINI_BATCH_SIZE*sizeof(float));
  }

  ACTIVATIONS_2D_TEST=(float**)malloc(NUM_LAYERS*sizeof(float*));
  for (i=1; i<NUM_LAYERS; i++) {
    ACTIVATIONS_2D_TEST[i]=(float*)malloc(SIZES[i]*TEST_SIZE*sizeof(float));
  }

  ZS_2D_TEST =(float**)malloc((NUM_LAYERS-1)*sizeof(float*));
  for (i=1; i<NUM_LAYERS; i++) {
    ZS_2D_TEST[i-1]=(float*)malloc(SIZES[i]*TEST_SIZE*sizeof(float));
  }

  // loop over epochs
  for (i=0; i<NUM_EPOCHS; i++) {

    //shuffle(TRAINING_DATA_X,TRAINING_DATA_Y,TRAINING_SIZE);

    // loop over mini-batches
    for (j=0; j<TRAINING_SIZE/MINI_BATCH_SIZE; j++) {

      J=j;

      update_mini_batch(&TRAINING_DATA_Y[j*MINI_BATCH_SIZE]);

    }

if (i == (NUM_EPOCHS-1)) {

    // test current weights and biases
    int total_correct=0,highest;

    ACTIVATIONS_2D_TEST[0]=TEST_DATA_X_2D;

    b_offset=w_offset=0;

    int m;

    for (k=1; k<NUM_LAYERS; k++) {

#pragma omp parallel for private(l,m)
      for (j=0; j<SIZES[k]; j++) {
        for (l=0; l<TEST_SIZE; l++) {
          ZS_2D_TEST[k-1][l*SIZES[k]+j]=0.0;
          for (m=0; m<SIZES[k-1]; m++) {
            ZS_2D_TEST[k-1][l*SIZES[k]+j]+=WEIGHTS[w_offset+(j*SIZES[k-1])+m]*ACTIVATIONS_2D_TEST[k-1][l*SIZES[k-1]+m];
          }
        }
      }
      for (j=0; j<SIZES[k]; j++) {
        for (l=0; l<TEST_SIZE; l++) {
          ZS_2D_TEST[k-1][l*SIZES[k]+j]+=BIASES[b_offset+j];
          ACTIVATIONS_2D_TEST[k][l*SIZES[k]+j]=sigmoid(ZS_2D_TEST[k-1][l*SIZES[k]+j]);
        }
      }
      b_offset+=SIZES[k];
      w_offset+=SIZES[k-1]*SIZES[k];
    }

    for (j=0; j<TEST_SIZE; j++) {

      highest=0;
      for (k=1; k<SIZES[NUM_LAYERS-1]; k++) {
        if (ACTIVATIONS_2D_TEST[NUM_LAYERS-1][j*SIZES[NUM_LAYERS-1]+k]>ACTIVATIONS_2D_TEST[NUM_LAYERS-1][j*SIZES[NUM_LAYERS-1]+highest]) {
          highest=k;
        }
      }

      if (highest==TEST_DATA_Y[j]) {
        total_correct++;
      }

    }

    //printf("%d %d %d\n",(int)i,total_correct,TEST_SIZE);

    PERCENT_CORRECT=(float)(total_correct)/(float)(TEST_SIZE);

}

  }

}

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
// Programmer: Mitch Horton
// Creation: February 26, 2015
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
// Creation: February 26, 2015
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{

  long i,j,k,l,m,n,b_offset,w_offset;
  NUM_LAYERS=3;
  SIZES=(int*)malloc(NUM_LAYERS*sizeof(int));
  SIZES[0]=784;
  SIZES[1]=30;
  SIZES[2]=10;
  NUM_EPOCHS=10;
  MINI_BATCH_SIZE=10;
  ETA=3.0;

  //printf("%d %d %f\n",NUM_EPOCHS,MINI_BATCH_SIZE,ETA);

  FILE *f_training_data_x = fopen("nn_data/training_data_x","r");
  FILE *f_training_data_y = fopen("nn_data/training_data_y","r");
  //FILE *f_validation_data_x = fopen("nn_data/validation_data_x","r");
  //FILE *f_validation_data_y = fopen("nn_data/validation_data_y","r");
  FILE *f_test_data_x = fopen("nn_data/test_data_x","r");
  FILE *f_test_data_y = fopen("nn_data/test_data_y","r");

  if(f_training_data_x == NULL)
  {
    printf("Input training file not found - please check data directory!\n");
    exit(0);
  }

  TRAINING_DATA_X_2D=(float*)malloc(TRAINING_SIZE*IMAGE_SIZE*sizeof(float));

  //TRAINING_DATA_X=(float**)malloc(TRAINING_SIZE*sizeof(float*));
  //for (i=0; i<TRAINING_SIZE; i++) {
    //TRAINING_DATA_X[i] = (float*)malloc(IMAGE_SIZE*sizeof(float));
  //}

  //float **validation_data_x=(float**)malloc(10000*sizeof(float*));
  //for (i=0; i<VALIDATION_SIZE; i++) {
    //validation_data_x[i] = (float*)malloc(IMAGE_SIZE*sizeof(float));
  //}

  TEST_DATA_X_2D=(float*)malloc(TEST_SIZE*IMAGE_SIZE*sizeof(float));

  //TEST_DATA_X=(float**)malloc(10000*sizeof(float*));
  //for (i=0; i<TEST_SIZE; i++) {
    //TEST_DATA_X[i] = (float*)malloc(IMAGE_SIZE*sizeof(float));
  //}

  TRAINING_DATA_Y = (int*)malloc(TRAINING_SIZE*sizeof(int));
  //int *validation_data_y = (int*)malloc(VALIDATION_SIZE*sizeof(int));
  TEST_DATA_Y = (int*)malloc(TEST_SIZE*sizeof(int));

  float total;

  for (i=0; i<TRAINING_SIZE; i++) {
    fread(&TRAINING_DATA_X_2D[i*IMAGE_SIZE],sizeof(float),IMAGE_SIZE,f_training_data_x);
  }
  fclose(f_training_data_x);

  //for (i=0; i<VALIDATION_SIZE; i++) {
    //fread(validation_data_x[i],sizeof(float),IMAGE_SIZE,f_validation_data_x);
  //}
  //fclose(f_validation_data_x);

  for (i=0; i<TEST_SIZE; i++) {
    fread(&TEST_DATA_X_2D[i*IMAGE_SIZE],sizeof(float),IMAGE_SIZE,f_test_data_x);
  }
  fclose(f_test_data_x);

  fread(TRAINING_DATA_Y,sizeof(int),TRAINING_SIZE,f_training_data_y);
  fclose(f_training_data_y);

  //fread(validation_data_y,sizeof(int),VALIDATION_SIZE,f_validation_data_y);
  //fclose(f_validation_data_y);

  fread(TEST_DATA_Y,sizeof(int),TEST_SIZE,f_test_data_y);
  fclose(f_test_data_y);

  total=0.0;
  for (i=0; i<TEST_SIZE; i++) {
    total+=(TEST_DATA_Y[i])*(TEST_DATA_Y[i]);
  }
  //printf("%f\n",sqrt(total));

  total=0.0;
  for (i=0; i<IMAGE_SIZE; i++) {
    total+=(TEST_DATA_X_2D[IMAGE_SIZE+i])*(TEST_DATA_X_2D[IMAGE_SIZE+i]);
  }
  //printf("%f\n",total/IMAGE_SIZE);

  //printf("%d %d\n",TEST_DATA_Y[0],TEST_DATA_Y[1]);

  TBS=0;
  TWS=0;

  for (i=1; i<NUM_LAYERS; i++) {
    TBS+=SIZES[i];
    TWS+=SIZES[i-1]*SIZES[i];
  }

  BIASES =(float*)malloc(TBS*sizeof(float));
  WEIGHTS=(float*)malloc(TWS*sizeof(float));

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

  //printf("%d %d\n",TBS,TWS);

  total=0.0;
  for (j=0; j<TRAINING_SIZE*IMAGE_SIZE; j++) {
    total+=TRAINING_DATA_X_2D[j]*TRAINING_DATA_X_2D[j];
  }
  //printf("%f\n",sqrt(total));

  //shuffle(TRAINING_DATA_X,TRAINING_DATA_Y,TRAINING_SIZE);

  total=0.0;
  for (j=0; j<TRAINING_SIZE*IMAGE_SIZE; j++) {
    total+=(TRAINING_DATA_X_2D[j])*(TRAINING_DATA_X_2D[j]);
  }
  //printf("%f\n",sqrt(total));

#pragma offload target(MIC) in(NUM_EPOCHS) \
                            in(I) \
                            in(J) \
                            in(BIASES:length(TBS) free_if(0)) \
                            in(WEIGHTS:length(TWS) free_if(0)) \
                            in(NABLA_B:length(TBS) free_if(0)) \
                            in(NABLA_W:length(TWS) free_if(0)) \
                            in(DELTA_NABLA_B:length(TBS) free_if(0)) \
                            in(DELTA_NABLA_W:length(TWS) free_if(0)) \
                            in(MINI_BATCH_SIZE) \
                            in(TBS) \
                            in(TWS) \
                            in(ETA) \
                            in(NUM_LAYERS) \
                            in(SIZES:length(NUM_LAYERS) free_if(0)) \
                            in(TRAINING_DATA_X_2D:length(TRAINING_SIZE*IMAGE_SIZE) free_if(0)) \
                            in(TRAINING_DATA_Y:length(TRAINING_SIZE) free_if(0)) \
                            in(TEST_DATA_X_2D:length(TEST_SIZE*IMAGE_SIZE) free_if(0)) \
                            in(TEST_DATA_Y:length(TEST_SIZE) free_if(0)) \
                            out(PERCENT_CORRECT)
  {
  }

  // Start the timer for the PCIe transfer
  int txToDevTimerHandle = Timer::Start();
#pragma offload target(MIC) in(NUM_EPOCHS) \
                            in(I) \
                            in(J) \
                            in(BIASES:length(TBS) alloc_if(0) free_if(0)) \
                            in(WEIGHTS:length(TWS) alloc_if(0) free_if(0)) \
                            in(NABLA_B:length(TBS) alloc_if(0) free_if(0)) \
                            in(NABLA_W:length(TWS) alloc_if(0) free_if(0)) \
                            in(DELTA_NABLA_B:length(TBS) alloc_if(0) free_if(0)) \
                            in(DELTA_NABLA_W:length(TWS) alloc_if(0) free_if(0)) \
                            in(MINI_BATCH_SIZE) \
                            in(TBS) \
                            in(TWS) \
                            in(ETA) \
                            in(NUM_LAYERS) \
                            in(SIZES:length(NUM_LAYERS) alloc_if(0) free_if(0)) \
                            in(TRAINING_DATA_X_2D:length(TRAINING_SIZE*IMAGE_SIZE) alloc_if(0) free_if(0)) \
                            in(TRAINING_DATA_Y:length(TRAINING_SIZE) alloc_if(0) free_if(0)) \
                            in(TEST_DATA_X_2D:length(TEST_SIZE*IMAGE_SIZE) alloc_if(0) free_if(0)) \
                            in(TEST_DATA_Y:length(TEST_SIZE) alloc_if(0) free_if(0)) \
                            out(PERCENT_CORRECT)
  {
  }
  double transfer_time = Timer::Stop(txToDevTimerHandle, "tx to dev");

  int iterations = 2;
  int first = 1;

  for (int i=0; i<iterations; i++) {

    // Time it takes for the actual neural net computation
    int kernelTimerHandle = Timer::Start();
#pragma offload target(MIC) nocopy(NUM_EPOCHS) \
                            nocopy(I) \
                            nocopy(J) \
                            nocopy(BIASES) \
                            nocopy(WEIGHTS) \
                            nocopy(NABLA_B) \
                            nocopy(NABLA_W) \
                            nocopy(DELTA_NABLA_B) \
                            nocopy(DELTA_NABLA_W) \
                            nocopy(MINI_BATCH_SIZE) \
                            nocopy(TBS) \
                            nocopy(TWS) \
                            nocopy(ETA) \
                            nocopy(NUM_LAYERS) \
                            nocopy(SIZES) \
                            nocopy(TRAINING_DATA_X_2D) \
                            nocopy(TRAINING_DATA_Y) \
                            nocopy(TEST_DATA_X_2D) \
                            nocopy(TEST_DATA_Y) \
                            nocopy(PERCENT_CORRECT)
    main_loop();
    double neuralnet_time = Timer::Stop(kernelTimerHandle, "neuralnet");

    if (first == 1) {

      first = 0;

      txToDevTimerHandle = Timer::Start();
#pragma offload target(MIC) nocopy(NUM_EPOCHS) \
                            nocopy(I) \
                            nocopy(J) \
                            nocopy(BIASES:length(TBS) alloc_if(0) free_if(0)) \
                            nocopy(WEIGHTS:length(TWS) alloc_if(0) free_if(0)) \
                            nocopy(NABLA_B:length(TBS) alloc_if(0) free_if(0)) \
                            nocopy(NABLA_W:length(TWS) alloc_if(0) free_if(0)) \
                            nocopy(DELTA_NABLA_B:length(TBS) alloc_if(0) free_if(0)) \
                            nocopy(DELTA_NABLA_W:length(TWS) alloc_if(0) free_if(0)) \
                            nocopy(MINI_BATCH_SIZE) \
                            nocopy(TBS) \
                            nocopy(TWS) \
                            nocopy(ETA) \
                            nocopy(NUM_LAYERS) \
                            nocopy(SIZES:length(NUM_LAYERS) alloc_if(0) free_if(0)) \
                            nocopy(TRAINING_DATA_X_2D:length(TRAINING_SIZE*IMAGE_SIZE) alloc_if(0) free_if(0)) \
                            nocopy(TRAINING_DATA_Y:length(TRAINING_SIZE) alloc_if(0) free_if(0)) \
                            nocopy(TEST_DATA_X_2D:length(TEST_SIZE*IMAGE_SIZE) alloc_if(0) free_if(0)) \
                            nocopy(TEST_DATA_Y:length(TEST_SIZE) alloc_if(0) free_if(0)) \
                            out(PERCENT_CORRECT)
      {
      }
      transfer_time += Timer::Stop(txToDevTimerHandle, "tx to dev");

    }

    if (PERCENT_CORRECT < .8) {
      return;
    }

    char atts[1024];
    sprintf(atts, "%d training_sets", TRAINING_SIZE*NUM_EPOCHS);
    resultDB.AddResult("Learning-Rate", atts, "training_sets/s", TRAINING_SIZE*NUM_EPOCHS / neuralnet_time);
    resultDB.AddResult("Learning-Rate_PCIe", atts, "training_sets/s", TRAINING_SIZE*NUM_EPOCHS / (transfer_time+neuralnet_time));

    //cout << transfer_time << " " << neuralnet_time << " " << PERCENT_CORRECT << endl;

  }

// Clean up MIC storage
#pragma offload target(MIC) in(NUM_EPOCHS) \
                            in(I) \
                            in(J) \
                            in(BIASES:length(TBS) alloc_if(0) ) \
                            in(WEIGHTS:length(TWS) alloc_if(0) ) \
                            in(NABLA_B:length(TBS) alloc_if(0) ) \
                            in(NABLA_W:length(TWS) alloc_if(0) ) \
                            in(DELTA_NABLA_B:length(TBS) alloc_if(0) ) \
                            in(DELTA_NABLA_W:length(TWS) alloc_if(0) ) \
                            in(MINI_BATCH_SIZE) \
                            in(TBS) \
                            in(TWS) \
                            in(ETA) \
                            in(NUM_LAYERS) \
                            in(SIZES:length(NUM_LAYERS) alloc_if(0) ) \
                            in(TRAINING_DATA_X_2D:length(TRAINING_SIZE*IMAGE_SIZE) alloc_if(0) ) \
                            in(TRAINING_DATA_Y:length(TRAINING_SIZE) alloc_if(0) ) \
                            in(TEST_DATA_X_2D:length(TEST_SIZE*IMAGE_SIZE) alloc_if(0) ) \
                            in(TEST_DATA_Y:length(TEST_SIZE) alloc_if(0) ) \
                            out(PERCENT_CORRECT)
  {
  }

  // Clean up host storage
  
  free(BIASES); 
  free(WEIGHTS);

  free(NABLA_B); 
  free(NABLA_W);

  free(DELTA_NABLA_B); 
  free(DELTA_NABLA_W);

  free(SIZES);

  free(TRAINING_DATA_X_2D);
  free(TRAINING_DATA_Y);

  free(TEST_DATA_X_2D);
  free(TEST_DATA_Y);

}

