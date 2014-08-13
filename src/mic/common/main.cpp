// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Kyle Spafford <kys@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011, UT-Battelle, LLC
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
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <string>
#include "omp.h"
//Allows for maintenance and management functionality
#include <miclib.h>
#include <stdio.h>
#include <string.h>

#include "offload.h"
#include "Timer.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "InvalidArgValue.h"

#ifdef  __MIC__ || __MIC2__
//#include <lmmintrin.h>
#include <pthread.h>
//#include <pthread_affinity_np.h>
#endif


// Forward Declarations
void addBenchmarkSpecOptions(OptionParser &op);
void RunBenchmark(OptionParser& op, ResultDatabase& resultDB);
int PrintDetailedMicInformation(struct mic_device *mdh);

//Separator for pretty print
const char *separator="\t---------------";

// ****************************************************************************
// Function: EnumerateDevicesAndChoose
//
// Purpose:
//   This function queries cuda about the available gpus in the system, prints
//   those results to standard out, and selects a device for use in the
//   benchmark.
//
// Arguments:
//   chosenDevice: logical number for the desired device
//   verbose: whether or not to print verbose output
//
// Returns:  Error or success code (1 or 0) to indicate selection of a valid device
//
// Programmer: Jeremy Meredith
// Creation:
//
// Modifications:
//   Jeremy Meredith, Tue Oct  9 17:27:04 EDT 2012
//   Added a windows-specific --noprompt, which unless the user passes it,
//   prompts the user to press enter before the program exits on Windows.
//   This is because on Windows, the console disappears when the program
//   exits, but our results go to the console.
//
//   Philip C. Roth, Wed Jul  3 14:03:42 EDT 2013
//   Adapted for Intel Xeon Phi (MIC) from CUDA version.
//
//   Jeffrey Young, Mon Aug 11 2014
//   Updated to make use of micmgmt functions to query the device.
//
// ****************************************************************************
  int
EnumerateDevicesAndChoose(int chosenDevice, bool verbose)
{

  //Code to use the micmgmt library is reused from the MPSS-included examples
  //which can be found in /usr/share/doc/micmgmt/examples/ for the Linux install
  int deviceCount, card_num, card;
  struct mic_devices_list *mdl;
  struct mic_device *mdh;
  int ret;
  uint32_t device_type;

  if (mic_get_devices(&mdl) != E_MIC_SUCCESS) {
    fprintf(stderr, "Failed to get cards list: %s: %s\n",
        mic_get_error_string(), strerror(errno));
    return 1;
  }

  //Could also use this line to get the number of devices
  //int deviceCount = _Offload_number_of_devices();
  if (mic_get_ndevices(mdl, &deviceCount) != E_MIC_SUCCESS) {
    fprintf(stderr, "Failed to get number of cards: %s: %s\n",
        mic_get_error_string(), strerror(errno));
    (void)mic_free_devices(mdl);
    return 1;
  }

  //Check to make sure there are valid MIC devices in the system
  if (deviceCount == 0) {
    fprintf(stderr, "No MIC card found\n");
    (void)mic_free_devices(mdl);
    return 1;
  }

  if (verbose)
  {
    cout << "Number of devices = " << deviceCount << endl << endl;
  }

  //Although the chosen device is already specified, the verbose flag will
  //enumerate the features of all devices in the system
  for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex)
  {
    //Get card at index device
    if (mic_get_device_at_index(mdl, deviceIndex, &card) != E_MIC_SUCCESS) {
      fprintf(stderr, "Failed to get card at index %d: %s: %s\n",
          deviceIndex, mic_get_error_string(), strerror(errno));
      mic_free_devices(mdl);
      return 1;
    }

    //Open card and get a valid pointer to mic_device struct
    if (mic_open_device(&mdh, card) != E_MIC_SUCCESS) {
      fprintf(stderr, "Failed to open card %d: %s: %s\n",
          card_num, mic_get_error_string(), strerror(errno));
      return 1;
    }

    if (mic_get_device_type(mdh, &device_type) != E_MIC_SUCCESS) {
      fprintf(stderr, "%s: Failed to get device type: %s: %s\n",
          mic_get_device_name(mdh),
          mic_get_error_string(), strerror(errno));
      (void)mic_close_device(mdh);
      return 1;
    }

    if (device_type != KNC_ID) {
      fprintf(stderr, "Unknown device Type: %u\n", device_type);
      (void)mic_close_device(mdh);
      return 1;
    }

    if(verbose)
    {
      std::cout << "Device " << mic_get_device_name(mdh) << ":\n";


      //Call a separate function to print out more detailed information using micmgmt 
      if (PrintDetailedMicInformation(mdh) != 0) {
        (void)mic_close_device(mdh);
        mic_free_devices(mdl);
        return 1;
      }

      //Print out OpenMP reported features for the selected device
      cout << "\tOpenMP Features: " <<endl<<separator<<endl;
      cout << "\t\tMax threads:  = " << omp_get_max_threads_target( TARGET_MIC, deviceIndex ) << '\n';
      cout << "\t\tNum procs:    = " << omp_get_num_procs_target( TARGET_MIC, deviceIndex ) << '\n';
      cout << "\t\tDynamic:      = " << omp_get_dynamic_target( TARGET_MIC, deviceIndex ) << '\n';
      cout << "\t\tNested:       = " << omp_get_nested_target( TARGET_MIC, deviceIndex ) << '\n';
      cout << endl<<endl;
    }

    //Close each device
    (void)mic_close_device(mdh);

  }
  //Free the device list
  (void)mic_free_devices(mdl);

  std::cout << "Chosen device:"
    << " index=" << chosenDevice
    << std::endl;

  return 0;
}


// ****************************************************************************
// Function: PrintDetailedMicInformation
//
// Purpose:
//   Prints detailed information about the selected MIC device using the micmgmt
//   library. This code is based on example code from the MPSS 2014 release and 
//   is located under /usr/share/doc/micmgmt/examples/ in Linux. For more
//   information on libmicmgmt, see the Intel Xeon Phi Coprocessor
//   System Software Developer's Guide, last updated in March 2014
//
// Returns:  Error or success code (1 or 0) to indicate selection of a valid device
//
// Arguments: mic_device struct
//
//
// Programmer: Jeffrey Young
// Creation: August 11 2014
//
// Modifications:
//
// ****************************************************************************



  int
PrintDetailedMicInformation(struct mic_device *mdh)
{
  uint32_t NAME_MAX = 1000;
  //Convert KB results to MB for readability
  uint32_t KB_to_MB = 2<<9;

  //Memory statistics variables
  struct mic_device_mem *minfo;
  char mem_type_str[NAME_MAX];
  size_t size;
  uint32_t density, msize, mem_freq;
  uint16_t ecc;

  /* Memory-related statistics */
  cout<<"\tMIC Memory: "<<endl<<separator<<endl;
  if (mic_get_memory_info(mdh, &minfo) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get memory "
        "information: %s: %s\n", mic_get_device_name(mdh),
        mic_get_error_string(), strerror(errno));
    return 1;
  }

  if (mic_get_memory_density(minfo, &density) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get memory density: "
        "%s: %s\n", mic_get_device_name(mdh),
        mic_get_error_string(), strerror(errno));
    (void)mic_free_memory_info(minfo);
    return 1;
  }
  printf("\t\tMemory density: %u Kbits/device\n", density);

  if (mic_get_memory_size(minfo, &msize) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get memory size: "
        "%s: %s\n", mic_get_device_name(mdh),
        mic_get_error_string(), strerror(errno));
    (void)mic_free_memory_info(minfo);
    return 1;
  }
  printf("\t\tMemory size: %u KB / %u MB\n", msize, msize/KB_to_MB);

  size = sizeof(mem_type_str);
  if (mic_get_memory_type(minfo, mem_type_str, &size) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get memory type: "
        "%s: %s\n", mic_get_device_name(mdh),
        mic_get_error_string(), strerror(errno));
    (void)mic_free_memory_info(minfo);
    return 1;
  }
  printf("\t\tMemory type: %s\n", mem_type_str);

  if (mic_get_memory_frequency(minfo, &mem_freq) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get memory frequency: "
        "%s: %s\n", mic_get_device_name(mdh),
        mic_get_error_string(), strerror(errno));
    (void)mic_free_memory_info(minfo);
    return 1;
  }
  printf("\t\tMemory frequency: %u KHz / %u MHz\n\n", mem_freq, (mem_freq/KB_to_MB)); 

  (void)mic_free_memory_info(minfo);

  /* Core Information */
  struct mic_cores_info *cinfo;
  uint32_t num_cores, core_freq;
  /* Core Utilization Metrics */
  struct mic_core_util *cutil = NULL;
  uint16_t  thread_core;

 
//Get the core information struct 
 if (mic_get_cores_info(mdh, &cinfo) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get core "
        "information: %s: %s\n", mic_get_device_name(mdh),
        mic_get_error_string(), strerror(errno));
    return 1;
  }

  //And also update the core utilities struct to get the number of threads
  if (mic_alloc_core_util(&cutil) != E_MIC_SUCCESS) {
    fprintf(
        stderr,
        "%s: Failed to allocate Core utilization information: %s: %s\n",
        mic_get_device_name(mdh),
        mic_get_error_string(),
        strerror(errno));
    return 1;
  }

  if (mic_update_core_util(mdh, cutil) != E_MIC_SUCCESS) {
    fprintf(
        stderr,
        "%s: Failed to update Core utilization information: %s: %s\n",
        mic_get_device_name(mdh),
        mic_get_error_string(),
        strerror(errno));
    (void)mic_free_core_util(cutil);
    return 1;
  }

  cout<<"\tCore Information: "<<endl<<separator<<endl;
  if (mic_get_cores_info(mdh, &cinfo) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get core "
        "information: %s: %s\n", mic_get_device_name(mdh),
        mic_get_error_string(), strerror(errno));
    return 1;
  }

  if (mic_get_cores_count(cinfo, &num_cores) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get number of cores: %s: %s\n",
        mic_get_device_name(mdh),
        mic_get_error_string(),
        strerror(errno));
    (void)mic_free_cores_info(cinfo);
    return 1;
  }
  printf("\t\tNumber of cores: %u\n", num_cores);

  //Placing number of threads here since it makes sene when correlated with number of cores
  if (mic_get_threads_core(cutil, &thread_core) != E_MIC_SUCCESS) {
    fprintf(
        stderr,
        "%s: Failed to get the Number of threads per core : %s: %s\n",
        mic_get_device_name(mdh),
        mic_get_error_string(),
        strerror(errno));
    (void)mic_free_core_util(cutil);
    return 1;
  }

  printf("\t\tNumber of threads per core: %u\n", thread_core); 
  printf("\t\tTotal MIC threads: %u\n", num_cores*thread_core);


  if (mic_get_cores_frequency(cinfo, &core_freq) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get core frequency: %s: %s\n",
        mic_get_device_name(mdh),
        mic_get_error_string(),
        strerror(errno));
    (void)mic_free_cores_info(cinfo);
    return 1;
  }
  printf("\t\tCore frequency: %u KHz / %3f GHz\n\n", core_freq,  ((double)core_freq)/((double)(KB_to_MB*KB_to_MB))); 

  (void)mic_free_cores_info(cinfo);
  (void)mic_free_core_util(cutil);


  /* Processor Information */
  struct mic_processor_info *pinfo;
  uint32_t id;
  char stepping[NAME_MAX];
  uint16_t model, model_ext, type;

  cout<<"\tProcessor Information: "<<endl<<separator<<endl;
  if (mic_get_processor_info(mdh, &pinfo) != E_MIC_SUCCESS) {
    fprintf(stderr,
        "%s: Failed to get processor information: %s: %s\n",
        mic_get_device_name(
          mdh),
        mic_get_error_string(),
        strerror(errno));
    return 1;
  }

  if (mic_get_processor_model(pinfo, &model,
        &model_ext) != E_MIC_SUCCESS) {
    fprintf(
        stderr,
        "%s: Failed to get processor model and processor extended model: %s: %s\n",
        mic_get_device_name(mdh),
        mic_get_error_string(),
        strerror(errno));
    (void)mic_free_processor_info(pinfo);
    return 2;
  }
  printf("\t\tProcessor model: %u\n", model);
  printf("\t\tProcessor model Extension: %u\n", model_ext);

  if (mic_get_processor_type(pinfo, &type) != E_MIC_SUCCESS) {
    fprintf(stderr, "%s: Failed to get processor type: %s: %s\n",
        mic_get_device_name(mdh),
        mic_get_error_string(),
        strerror(errno));
    (void)mic_free_processor_info(pinfo);
    return 3;
  }
  printf("\t\tProcessor type: %u\n", type);

  if (mic_get_processor_steppingid(pinfo, &id) != E_MIC_SUCCESS) {
    fprintf(stderr,
        "%s: Failed to get processor stepping id: %s: %s\n",
        mic_get_device_name(
          mdh),
        mic_get_error_string(),
        strerror(errno));
    (void)mic_free_processor_info(pinfo);
    return 4;
  }
  printf("\t\tProcessor stepping id: %u\n", id);

  size = sizeof(stepping);
  if (mic_get_processor_stepping(pinfo, stepping,
        &size) != E_MIC_SUCCESS) {
    fprintf(stderr,
        "%s: Failed to get processor stepping : %s: %s\n",
        mic_get_device_name(
          mdh),
        mic_get_error_string(),
        strerror(errno));
    (void)mic_free_processor_info(pinfo);
    return 5;
  }
  printf("\t\tProcessor stepping: %s\n\n", stepping);

  (void)mic_free_processor_info(pinfo);

  return 0;
}


// ****************************************************************************
// Function: main
//
// Purpose:
//   The main function takes care of initialization (device and MPI),  then
//   performs the benchmark and prints results.
//
// Arguments:
//
//
// Programmer: Jeremy Meredith
// Creation:
//
// Modifications:
//   Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//   Split timing reports into detailed and summary.  For serial code, we
//   report all trial values, and for parallel, skip the per-process vals.
//   Also detect and print outliers from parallel runs.
//
//   Philip Roth, Wed Jul  3 14:00:12 EDT 2013
//   Adapted for Intel Xeon Phi offload programming model.
//
// ****************************************************************************
  int
main(int argc, char *argv[])
{
  int ret = 0;
  bool noprompt = false;

  try
  {
#ifdef PARALLEL
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cerr << "MPI Task " << rank << "/" << size - 1 << " starting....\n";
#endif

    // Get args
    OptionParser op;

    //Add shared options to the parser
    op.addOption("device", OPT_VECINT, "0",
        "specify device(s) to run on", 'd');
    op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
    op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
    op.addOption("size", OPT_INT, "1", "specify problem size", 's');
    op.addOption("infoDevices", OPT_BOOL, "",
        "show info for available platforms and devices", 'i');
#ifdef _WIN32
    op.addOption("noprompt", OPT_BOOL, "", "don't wait for prompt at program exit");
#endif

    addBenchmarkSpecOptions(op);

    if (!op.parse(argc, argv))
    {
#ifdef PARALLEL
      if (rank == 0)
        op.usage();
      MPI_Finalize();
#else
      op.usage();
#endif
      return (op.HelpRequested() ? 0 : 1);
    }

    bool verbose = op.getOptionBool("verbose");
    bool infoDev = op.getOptionBool("infoDevices");
#ifdef _WIN32
    noprompt = op.getOptionBool("noprompt");
#endif

    int device;
#ifdef PARALLEL
    NodeInfo ni;
    int myNodeRank = ni.nodeRank();
    vector<long long> deviceVec = op.getOptionVecInt("device");
    if (myNodeRank >= deviceVec.size()) {
      // Default is for task i to test device i
      device = myNodeRank;
    } else {
      device = deviceVec[myNodeRank];
    }
#else
    device = op.getOptionVecInt("device")[0];
#endif
    int deviceCount = _Offload_number_of_devices();
    if (device >= deviceCount) {
      cerr << "Warning: device index: " << device <<
        " out of range, defaulting to device 0.\n";
      device = 0;
    }

    // Initialization - check to make sure a device was selected
    // Print detailed statistics if infoDev is true
    if(EnumerateDevicesAndChoose(device, infoDev))
    {
      cerr << "No device could be opened - exiting"<<endl;
      return 1;
    }

    ResultDatabase resultDB;

    // Run the benchmark
    RunBenchmark(op, resultDB);

#ifndef PARALLEL
    resultDB.DumpDetailed(cout);
#else
    ParallelResultDatabase pardb;
    pardb.MergeSerialDatabases(resultDB,MPI_COMM_WORLD);
    if (rank==0)
    {
      pardb.DumpSummary(cout);
      pardb.DumpOutliers(cout);
    }
#endif

  }
  catch( InvalidArgValue& e )
  {
    std::cerr << e.what() << ": " << e.GetMessage() << std::endl;
    ret = 1;
  }
  catch( std::exception& e )
  {
    std::cerr << e.what() << std::endl;
    ret = 1;
  }
  catch( ... )
  {
    ret = 1;
  }


#ifdef PARALLEL
  MPI_Finalize();
#endif

#ifdef _WIN32
  if (!noprompt)
  {
    cout << "Press return to exit\n";
    cin.get();
  }
#endif

  return ret;
}

