#!/usr/bin/perl
use strict;

my $SHOC_VERSION = "1.1.5";

# ----------------------------------------------------------------------------
#  Output for the .csv results file
# ----------------------------------------------------------------------------

# This lists the names of the results in the order they need to be listed
# in the results.csv file.  Note that the names of these results must match
# the result names in the benchmark lists below.
my @CSVResults = (
"maxspflops",
"maxdpflops",
"gmem_readbw",
"gmem_writebw",
"gmem_readbw_strided",
"gmem_writebw_strided",
"lmem_readbw",
"lmem_writebw",
"tex_readbw",
"bspeed_download",
"bspeed_readback",
"fft_sp",
"ifft_sp",
"fft_sp_pcie",
"ifft_sp_pcie",
"fft_dp",
"ifft_dp",
"fft_dp_pcie",
"ifft_dp_pcie",
"sgemm_n",
"sgemm_t",
"sgemm_n_pcie",
"sgemm_t_pcie",
"dgemm_n",
"dgemm_t",
"dgemm_n_pcie",
"dgemm_t_pcie",
"md_sp_flops",
"md_sp_bw",
"md_sp_flops_pcie",
"md_sp_bw_pcie",
"md_dp_flops",
"md_dp_bw",
"md_dp_flops_pcie",
"md_dp_bw_pcie",
"reduction",
"reduction_pcie",
"reduction_dp",
"reduction_dp_pcie",
"scan",
"scan_pcie",
"scan_dp",
"scan_dp_pcie",
"sort",
"sort_pcie",
"spmv_csr_scalar_sp",
"spmv_csr_vector_sp",
"spmv_ellpackr_sp",
"spmv_csr_scalar_dp",
"spmv_csr_vector_dp",
"spmv_ellpackr_dp",
"stencil",
"stencil_dp",
"s3d",
"s3d_pcie",
"s3d_dp",
"s3d_dp_pcie",
"triad_bw",
"ocl_kernel",
"ocl_queue",
"bfs",
"bfs_pcie",
"bfs_teps",
"md5hash",
"nn_learning",
"nn_learning_pcie"
#"qtc",
#"qtc_kernel"
);

# ----------------------------------------------------------------------------
#  Actual benchmarks to run and results to extract (serial only)
# ----------------------------------------------------------------------------

# Note that the names of these results must match
# the result names in the CSV output list above.

# Here is the structure of the benchmark arrays:
# ["BenchmarkProgram", iscuda, isopencl, istp,
#        ["firstresultname", findmin/findmax/etc, matchtext],
#        ["secondresultname", .....] ],
# ["NextBenchmark", ........ ]

# 
# Note that all fields must be present in each item the Parallel and Serial 
# benchmark arrays.  In particular, you cannot add a field to one of the
# arrays without adding it to the other.  So, even though 'istp' does not
# mean anything in the context of serial benchmarks, we must have it.
#


my @SerialBenchmarks = (
[ "BusSpeedDownload",  1, 1, 0,
  ["bspeed_download",             \&findmax,     "DownloadSpeed"]
],
[ "BusSpeedReadback",  1, 1, 0,
  ["bspeed_readback",             \&findmax,     "ReadbackSpeed"]
],
[ "MaxFlops",          1, 1, 0,
  ["maxspflops",                  \&findanymax,  "-SP"],
  ["maxdpflops",                  \&findanymax,  "-DP"]
],
[ "DeviceMemory",      1, 1, 0,
  ["gmem_readbw",                 \&findmax,     "readGlobalMemoryCoalesced"],
  ["gmem_readbw_strided",         \&findmax,     "readGlobalMemoryUnit"],
  ["gmem_writebw",                \&findmax,     "writeGlobalMemoryCoalesced"],
  ["gmem_writebw_strided",        \&findmax,     "writeGlobalMemoryUnit"],
  ["lmem_readbw",                 \&findmax,     "readLocalMemory"],
  ["lmem_writebw",                \&findmax,     "writeLocalMemory"],
  ["tex_readbw",                  \&findmax,     "TextureRepeatedRandomAccess"]
],
[ "KernelCompile",     0, 1, 0, # OpenCL-only
  ["ocl_kernel",                  \&findmin,     "BuildProgram"]
],
[ "QueueDelay",        0, 1, 0, # OpenCL-only
  ["ocl_queue",                   \&findmin,     "SSDelay"]
],
[ "BFS",               1, 1, 0,
  ["bfs",                         \&findmax,     "BFS"],
  ["bfs_pcie",                    \&findmax,     "BFS_PCIe"],
  ["bfs_teps",                    \&findmax,     "BFS_teps"]
],

[ "FFT",               1, 1, 0,
  ["fft_sp",                      \&findmax,     "SP-FFT"],
  ["fft_sp_pcie",                 \&findmax,     "SP-FFT_PCIe"],
  ["ifft_sp",                     \&findmax,     "SP-FFT-INV"],
  ["ifft_sp_pcie",                \&findmax,     "SP-FFT-INV_PCIe"],
  ["fft_dp",                      \&findmax,     "DP-FFT"],
  ["fft_dp_pcie",                 \&findmax,     "DP-FFT_PCIe"],
  ["ifft_dp",                     \&findmax,     "DP-FFT-INV"],
  ["ifft_dp_pcie",                \&findmax,     "DP-FFT-INV_PCIe"]
],
[ "GEMM",             1, 1, 0,
  ["sgemm_n",                     \&findmax,     "SGEMM-N"],
  ["sgemm_t",                     \&findmax,     "SGEMM-T"],
  ["sgemm_n_pcie",                \&findmax,     "SGEMM-N_PCIe"],
  ["sgemm_t_pcie",                \&findmax,     "SGEMM-T_PCIe"],
  ["dgemm_n",                     \&findmax,     "DGEMM-N"],
  ["dgemm_t",                     \&findmax,     "DGEMM-T"],
  ["dgemm_n_pcie",                \&findmax,     "DGEMM-N_PCIe"],
  ["dgemm_t_pcie",                \&findmax,     "DGEMM-T_PCIe"]
],
[ "MD",                1, 1, 0,
  ["md_sp_flops",                 \&findmax,     "MD-LJ"],
  ["md_sp_bw",                    \&findmax,     "MD-LJ-Bandwidth"],
  ["md_sp_flops_pcie",            \&findmax,     "MD-LJ_PCIe"],
  ["md_sp_bw_pcie",               \&findmax,     "MD-LJ-Bandwidth_PCIe"],
  ["md_dp_flops",                 \&findmax,     "MD-LJ-DP"],
  ["md_dp_bw",                    \&findmax,     "MD-LJ-DP-Bandwidth"],
  ["md_dp_flops_pcie",            \&findmax,     "MD-LJ-DP_PCIe"],
  ["md_dp_bw_pcie",               \&findmax,     "MD-LJ-DP-Bandwidth_PCIe"]
],
[ "MD5Hash",           1, 1, 0,
  ["md5hash",                     \&findmax,     "MD5Hash"]
],
[ "NeuralNet",         1, 0, 0,
  ["nn_learning",                 \&findmean,    "Learning-Rate"],
  ["nn_learning_pcie",            \&findmean,    "Learning-Rate_PCIe"]
],
[ "Reduction",         1, 1, 0,
  ["reduction",                   \&findmax,     "Reduction"],
  ["reduction_pcie",              \&findmax,     "Reduction_PCIe"],
  ["reduction_dp",                \&findmax,     "Reduction-DP"],
  ["reduction_dp_pcie",           \&findmax,     "Reduction-DP_PCIe"]
],
[ "Scan",              1, 1, 0,
  ["scan",                        \&findmax,     "Scan"],
  ["scan_pcie",                   \&findmax,     "Scan_PCIe"],
  ["scan_dp",                     \&findmax,     "Scan-DP"],
  ["scan_dp_pcie",                \&findmax,     "Scan-DP_PCIe"]
],
[ "Sort",              1, 1, 0,
  ["sort",                        \&findmax,     "Sort-Rate"],
  ["sort_pcie",                   \&findmax,     "Sort-Rate_PCIe"]
],
[ "Spmv",              1, 1, 0,
  ["spmv_csr_scalar_sp",          \&findmax,     "CSR-Scalar-SP"],
  ["spmv_csr_scalar_sp_pcie",     \&findmax,     "CSR-Scalar-SP_PCIe"],
  ["spmv_csr_scalar_dp",          \&findmax,     "CSR-Scalar-DP"],
  ["spmv_csr_scalar_dp_pcie",     \&findmax,     "CSR-Scalar-DP_PCIe"],
  ["spmv_csr_scalar_pad_sp",      \&findmax,     "Padded_CSR-Scalar-SP"],
  ["spmv_csr_scalar_pad_sp_pcie", \&findmax,     "Padded_CSR-Scalar-SP_PCIe"],
  ["spmv_csr_scalar_pad_dp",      \&findmax,     "Padded_CSR-Scalar-DP"],
  ["spmv_csr_scalar_pad_dp_pcie", \&findmax,     "Padded_CSR-Scalar-DP_PCIe"],
  ["spmv_csr_vector_sp",          \&findmax,     "CSR-Vector-SP"],
  ["spmv_csr_vector_sp_pcie",     \&findmax,     "CSR-Vector-SP_PCIe"],
  ["spmv_csr_vector_dp",          \&findmax,     "CSR-Vector-DP"],
  ["spmv_csr_vector_dp_pcie",     \&findmax,     "CSR-Vector-DP_PCIe"],
  ["spmv_csr_vector_pad_sp",      \&findmax,     "Padded_CSR-Vector-SP"],
  ["spmv_csr_vector_pad_sp_pcie", \&findmax,     "Padded_CSR-Vector-SP_PCIe"],
  ["spmv_csr_vector_pad_dp",      \&findmax,     "Padded_CSR-Vector-DP"],
  ["spmv_csr_vector_pad_dp_pcie", \&findmax,     "Padded_CSR-Vector-DP_PCIe"],
  ["spmv_ellpackr_sp",            \&findmax,     "ELLPACKR-SP"],
  ["spmv_ellpackr_dp",            \&findmax,     "ELLPACKR-DP"]
],
[ "Stencil2D",         1, 1, 0,
  ["stencil",                     \&findmax,     "SP_Sten2D"],
  ["stencil_dp",                  \&findmax,     "DP_Sten2D"]
],
[ "Triad",             1, 1, 0,
  ["triad_bw",                    \&findmax,     "TriadBdwth"]
],
[ "S3D",               1, 1, 0,
  ["s3d",                         \&findmax,     "S3D-SP"],
  ["s3d_pcie",                    \&findmax,     "S3D-SP_PCIe"],
  ["s3d_dp",                      \&findmax,     "S3D-DP"],
  ["s3d_dp_pcie",                 \&findmax,     "S3D-DP_PCIe"]
]#,
#[ "QTC",               1, 0, 1,
#  ["qtc",                         \&findmin,     "QTC+PCI_Trans."],
#  ["qtc_kernel",                  \&findmin,     "QTC_Kernel"]
#]
);

# ----------------------------------------------------------------------------
#  Actual benchmarks to run and results to extract (parallel only)
# ----------------------------------------------------------------------------

# Note that the names of these results must match
# the result names in the CSV output list above.

my @ParallelBenchmarks = (
[ "BusSpeedDownload",  1, 1, 0,
  ["bspeed_download",             \&findmean,    "DownloadSpeed(max)"]
],
[ "BusSpeedReadback",  1, 1, 0,
  ["bspeed_readback",             \&findmean,    "ReadbackSpeed(max)"]
],
[ "MaxFlops",          1, 1, 0,
  ["maxspflops",                  \&findanymean, "-SP\\\(max\\\)"],
  ["maxdpflops",                  \&findanymean, "-DP\\\(max\\\)"]
],
[ "DeviceMemory",      1, 1, 0,
  ["gmem_readbw",                 \&findmean,     "readGlobalMemoryCoalesced(max)"],
  ["gmem_readbw_strided",         \&findmean,     "readGlobalMemoryUnit(max)"],
  ["gmem_writebw",                \&findmean,     "writeGlobalMemoryCoalesced(max)"],
  ["gmem_writebw_strided",        \&findmean,     "writeGlobalMemoryUnit(max)"],
  ["lmem_readbw",                 \&findmean,     "readLocalMemory(max)"],
  ["lmem_writebw",                \&findmean,     "writeLocalMemory(max)"],
  ["tex_readbw",                  \&findmean,     "TextureRepeatedRandomAccess(max)"]
],
[ "KernelCompile",     0, 1, 0,
  ["ocl_kernel",                  \&findmean,    "BuildProgram(min)"]
],
[ "QueueDelay",        0, 1, 0,
  ["ocl_queue",                   \&findmean,    "SSDelay(min)"]
],
[ "FFT",               1, 1, 0,
  ["fft_sp",                      \&findmean,    "SP-FFT(max)"],
  ["fft_dp",                      \&findmean,    "DP-FFT(max)"]
],
[ "GEMM",             1, 1, 0,
  ["sgemm_n",                     \&findmean,    "SGEMM-N(max)"],
  ["dgemm_n",                     \&findmean,    "DGEMM-N(max)"]
],
[ "MD",                1, 1, 0,
  ["md_sp_flops",                 \&findmean,    "MD-LJ(max)"],
  ["md_dp_flops",                 \&findmean,    "MD-LJ-DP(max)"]
],
[ "MD5Hash",           1, 1, 0,
  ["md5hash",                     \&findmean,    "MD5Hash(max)"]
],
[ "Reduction",         1, 1, 0,
  ["reduction",                   \&findmean,    "Reduction(max)"],
  ["reduction_dp",                \&findmean,    "Reduction-DP(max)"]
],
[ "Scan",              1, 1, 0,
  ["scan",                        \&findmean,    "Scan(max)"],
  ["scan_dp",                     \&findmean,    "Scan-DP(max)"]
],
[ "Sort",              1, 1, 0,
  ["sort",                        \&findmean,    "Sort-Rate(max)"]
],
[ "Spmv",              1, 1, 0,
  ["spmv_csr_scalar_sp",          \&findmean,    "CSR-Scalar-SP(max)"],
  ["spmv_csr_vector_sp",          \&findmean,    "CSR-Vector-SP(max)"],
  ["spmv_ellpackr_sp",            \&findmean,    "ELLPACKR-SP(max)"],
  ["spmv_csr_scalar_dp",          \&findmean,    "CSR-Scalar-DP(max)"],
  ["spmv_csr_vector_dp",          \&findmean,    "CSR-Vector-DP(max)"],
  ["spmv_ellpackr_dp",            \&findmean,    "ELLPACKR-DP(max)"]
],
[ "Stencil2D",         1, 1, 1,
  ["stencil",                     \&findmean,     "SP_Sten2D(max)"],
  ["stencil_dp",                  \&findmean,     "DP_Sten2D(max)"]
],
[ "Triad",             1, 1, 0,
  ["triad_bw",                    \&findmean,    "TriadBdwth(max)"]
],
[ "S3D",               1, 1, 0,
  ["s3d",                         \&findmean,    "S3D-SP(max)"],
  ["s3d_dp",                      \&findmean,    "S3D-DP(max)"]
],
[ "QTC",                1, 0, 1,
  ["qtc",                         \&findmin,     "QTC+PCI_Trans.(min)"],
  ["qtc_kernel",                  \&findmin,     "QTC_Kernel(min)"]
]
);

# ----------------------------------------------------------------------------
#  Parse the arguments
# ----------------------------------------------------------------------------

# default arguments and other values
my $numNodes   = 1;
my $deviceList = "";
my $platformList = "";
my $numDevices = 1;
my $mode       = "";
my $sizeClass  = 1;
my $bindir     = "./bin";
my $readonly   = 0;
my $hostfile   = "";
my $singlebench= "";

# parse arguments
while (scalar(@ARGV) > 0) {
    my $arg = shift @ARGV;
    if ($arg eq "-h" or $arg eq "-help" or $arg eq "--help")
    {
        usage();
    }
    elsif ($arg eq "-d")
    {
        $deviceList = shift;
        die "-d argument requires a value\n" if (!defined $deviceList);
        my @tmp = split(/,/, $deviceList);
        foreach (@tmp)
        {
            die "Expected a comma-separated list of device indices for '-d' argument\n"
              if (! m/^\d+$/);
        }
        $numDevices = scalar(@tmp);
    }
    elsif ($arg eq "-p")
    {
        $platformList = shift;
        die "-p argument requires a value\n" if (!defined $platformList);
        my @tmp = split(/,/, $platformList);
        foreach (@tmp)
        {
            die "Expected a platform index for '-p' argument\n"
            if (! m/^\d+$/);
        }
    }
    elsif ($arg eq "-n")
    {
        $numNodes = shift;
        die "Expected a positive integer for -n argument.\n" if ($numNodes < 1);
    }
    elsif ($arg eq "-s")
    {
        $sizeClass = shift;
        die "-s argument requires a value\n" if (!defined $sizeClass);
        die "Expected a size class between 1 and 4 (e.g. -s 1)\n"
          if ($sizeClass <1 || $sizeClass > 4);
    }
    elsif ($arg eq "-bindir")
    {
        $bindir = shift;
        die "-bindir argument requires a value\n" if (!defined $bindir);
    }
    elsif ($arg eq "-hostfile")
    {
        $hostfile = shift;
        die "-hostfile argument requires a value\n" if (!defined $hostfile);
    }
    elsif ($arg eq "-cuda")
    {
        die "Please choose one of '-cuda' or '-opencl' to set the operating mode.\n"
          if ($mode ne "");
        $mode = "cuda";
    }
    elsif ($arg eq "-opencl")
    {
        die "Please choose one of '-cuda' or '-opencl' to set the operating mode.\n"
          if ($mode ne "");
        $mode = "opencl";
    }
    elsif ($arg eq "-benchmark")
    {
        $singlebench = shift;
        die "-benchmark argument requires a value\n" if (!defined $singlebench);
    }
    elsif ($arg eq "-read-only")
    {
        $readonly = 1;
    }
    else
    {
        print "Unexpected argument: '$arg'\n\n";
        usage();
    }
}

# test binary directory
if (! -d "$bindir/Serial")
{
    die "The directory $bindir doesn't appear to be the SHOC binary directory.\n" .
        "Either run from the SHOC install root directory or use the -bindir argument.\n";
}

# Check if there are executables in the binary directory.
# Note: this check is not exhaustive.
if ( ! (( -f "$bindir/Serial/OpenCL/Sort" && -x "$bindir/Serial/OpenCL/Sort" ) ||
        ( -f "$bindir/Serial/CUDA/Sort" && -x "$bindir/Serial/CUDA/Sort" )) )
    
{
    die "The SHOC benchmark programs are not present in $bindir.\n" .
        "Be sure that you have configured, built, and installed SHOC\n" .
        "(using the traditional GNU-style \"configure; make; make install\"\n" .
        "sequence of commands) and/or check your -bindir argument.\n";
}

# test cuda vs opencl
if ($mode eq "")
{
    die "Please choose either -cuda or -opencl.\n";
}

# ----------------------------------------------------------------------------
#  Set up logging and directories
# ----------------------------------------------------------------------------

# Create a directory to save logs for the benchmarks
if (-d "./Logs") {
    my $tmp = system"echo \"SHOC Version ${SHOC_VERSION}\">Logs/version.txt ";
}
else {
   my $retval = system("mkdir Logs");
   die "Unable to create logs directory" unless $retval == 0;
}

# try to duplicate stdout to a log file
open(SAVEOUT, ">&STDOUT");
if (! open(STDOUT, "| tee -i Logs/shoc${deviceList}.log"))
{
    open(STDOUT, ">&SAVEOUT");
    print "Warning: could not save screen output to log file.\n";
}

# ----------------------------------------------------------------------------
#  Final initialization
# ----------------------------------------------------------------------------

# Simple Script To Run SHOC Benchmarks
print "--- Welcome To The SHOC Benchmark Suite version ${SHOC_VERSION} --- \n";

# Find out the hostname
my $host_name = `hostname`;
chomp($host_name);
print "Hostname: $host_name \n";

# Print info about available devices and platforms
if ($platformList eq "")
{
    print "Platform selection not specified, default to platform #0\n";
    $platformList = "0";
}
else
{
    print "Specified platform IDs: $platformList\n";
}

printDevInfo($mode);
if ($deviceList eq "")
{
   print "Device selection not specified: defaulting to device #0.\n";   
   $numDevices = 1;
   $deviceList = "0";
}
else
{
    print "Specified $numDevices device IDs: $deviceList\n";
}

# Print the size class
print "Using size class: $sizeClass\n";

print "\n--- Starting Benchmarks ---\n";

# ----------------------------------------------------------------------------
#  Run the benchmarks!
# ----------------------------------------------------------------------------
my $numTasks = $numDevices * $numNodes;
my $benchmarks;
if ($numTasks == 1)
{
    $benchmarks = \@SerialBenchmarks;
}
else
{
    $benchmarks = \@ParallelBenchmarks;
}

# run the benchmarks and collect results
my %results;
foreach my $bench (@$benchmarks)
{
    my $program = $$bench[0];
    my $incuda  = $$bench[1];
    my $inopencl= $$bench[2];
    my $istp    = $$bench[3];

    # check if they specified a single benchmark before proceeding
    next if (($singlebench ne "") and
             ($program ne $singlebench));


    if ((!$incuda   and ($mode eq "cuda")) or
        (!$inopencl and ($mode eq "opencl")))
    {
        print "Skipping non-$mode benchmark $program\n";
        next;
    }

    my $logbase = buildFileName($program, $deviceList);
    my $command = buildCommand($program, $logbase, $istp);
    my $outfile = "${logbase}.log";

    if ($readonly)
    {
        print "Reusing previous results from $program\n";
    }
    else
    {
        print "Running benchmark $program\n";
        if ($command ne "" )
        {
            my $rc = system($command);
            my $q = $?;
            if ($q == -1)
            {
                printf "failed to execute: $!\n";
            }
            elsif( $? & 127 ) 
            {
                my $estr = sprintf "terminated with signal %d; exiting\n", ($q & 127);
                die $estr
            }
        }
    }

    my $numResults = ($#$bench) - 3;
    for (my $r=0; $r<$numResults; $r++)
    {
        my $res = $$bench[4+$r];
        my $resname = $$res[0];
        my $resfunc = $$res[1];
        my $respatt = $$res[2];

        my ($value, $units) = $resfunc->($outfile, $respatt);

        $results{$resname} = $value;
        print sprintf("%-45s","    result for $resname: ");
        if ($value eq "NoResult" or $value eq "BenchmarkError")
        {
            print "   $value\n";
        }
        else
        {
            print sprintf("% 10.4f $units\n",$value);
        }
    }
}

# print the results to the .csv file
open(OUTFILE, ">results.csv") or die $!;
my $numOutputs = scalar(@CSVResults);
print OUTFILE join(",",@CSVResults);
print OUTFILE "\n";
for (my $r=0; $r<$numOutputs; $r++)
{
    my $res = $results{$CSVResults[$r]};
    print OUTFILE "," if ($r != 0);
    print OUTFILE $res if (defined $res);
}
print OUTFILE "\n";
close(OUTFILE);


# done!
close(STDOUT);

# ----------------------------------------------------------------------------
#  Support subroutines
# ----------------------------------------------------------------------------

# Subroutine: findmin(fileName, testName)
# Purpose: Parses resultDB output to find minimum value for a specified test
# fileName -- name of log file to open
# testName -- name of test to look for
sub findmin {

   my $unit = "";
   my $filename = $_[0];
   my $testname = $_[1];

   open( LOGFILE, $filename );
   my $best = 1E+37;    # Arbitrary Large Number

   my $line;
   my @tokens;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $tokens[0] eq $testname ) {
         $unit = $tokens[2];
         if ( $tokens[6] < $best ) {    # min is the 6th column
            $best = $tokens[6];
         }
      }
   }
   close(LOGFILE);
   return (checkError($best), $unit);
}

# Subroutine: findmax(fileName, testName)
# Purpose: Parses resultDB output to find maximum value for a specified test
# fileName -- name of log file to open
# testName -- name of test to look for
sub findmax {

   my $unit = "";
   my $filename = $_[0];
   my $testname = $_[1];

   open( LOGFILE, $filename );
   my $best = -1;

   my $line;
   my @tokens;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $tokens[0] eq $testname ) {
         $unit = $tokens[2];
                        # We assume that the 7th column is the max (which could be
                        # nan or inf) and after that there are all the trials.
         for(my $i=7; $i<=$#tokens; $i++){
             if ( $tokens[$i] =~ /inf/ || $tokens[$i] =~ /nan/ ){
                 next;
             }
             if ( $tokens[$i] > $best ) {
                $best = $tokens[$i];
             }
         }
      }
   }
   close(LOGFILE);
   return (checkError($best), $unit);
}

# Subroutine: findanymax(fileName)
# Purpose: Parses resultDB output to find maximum value for any test
# fileName -- name of log file to open
sub findanymax {

   my $unit = "";
   my $filename = $_[0];
   my $pattern = $_[1];
   open( LOGFILE, $filename );
   my $best = -1;

   my $line;
   my @tokens;
   my $header_found = 0;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $header_found eq 1 && $tokens[0] =~ /$pattern/ ) {
         $unit = $tokens[2];
         if ( $tokens[7] > $best ) {
            $best = $tokens[7];
         }
      }
      if ( $tokens[0] eq "test" ) {
         $header_found = 1;
      }
   }
   close(LOGFILE);
   return (checkError($best), $unit);
}

# Subroutine: findmean(fileName, testName)
# Purpose: Parses resultDB output to find mean value for a specified test
# fileName -- name of log file to open
# testName -- name of test to look for
sub findmean {

   my $unit = "";
   my $filename = $_[0];
   my $testname = $_[1];
   open( LOGFILE, $filename );
   my $best = -1;
   my $line;
   my @tokens;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $tokens[0] eq $testname ) {
         $unit = $tokens[2];
         if ( $tokens[4] > $best ) {
            $best = $tokens[4];
         }
      }
   }
   close(LOGFILE);
   return (checkError($best), $unit);
}

sub findanymean {

   my $unit = "";
   my $filename = $_[0];
   my $pattern = $_[1];
   open( LOGFILE, $filename );
   my $best = -1;

   my $line;
   my @tokens;
   my $header_found = 0;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $header_found eq 1 && $tokens[0] =~ /$pattern/ ) {
         $unit = $tokens[2];
         if ( $tokens[4] > $best ) {
            $best = $tokens[4];
         }
      }
      if ( $tokens[0] eq "test" ) {
         $header_found = 1;
      }
   }
   close(LOGFILE);
   return (checkError($best), $unit);
}

# Subroutine: buildCommand(testName, deviceNum, istp)
# Purpose: Helper routine to construct commands to run benchmarks
# testName -- name of test
# deviceNum -- device number
# istp -- for parallel benchmarks, is it TP or EP?

sub buildCommand {
   my $progname = $_[0];
   my $logbase  = $_[1];
   my $istp     = $_[2];

   my $command = "";

   # Parallel; start with mpirun and maybe a hostfile
   if ($numTasks > 1)
   {
       $command .= "mpirun -np $numTasks ";
       $command .= "-hostfile $hostfile " if ($hostfile ne "");
   }

   # Construct the program path
   $command .= $bindir;
   $command .= "/Serial"  if ($numTasks == 1);
   if ($numTasks > 1)
   {
       if ($istp == 1 )
       {
           $command .= "/TP"
       }
       else
       {
           $command .= "/EP"
       }
   }
   $command .= "/OpenCL/" if ($mode eq "opencl");
   $command .= "/CUDA/"   if ($mode eq "cuda");
   $command .= $progname;

   # Add size and maybe device options
   $command .= " -s $sizeClass";
   $command .= " -p $platformList" if($platformList ne "" && $mode eq "opencl");
   $command .= " -d $deviceList" if ($deviceList ne "");

   # Output redirection
   $command .= " > " . $logbase . ".log";
   $command .= " 2> ". $logbase . ".err";

   # print "Built command: $command\n";
   return $command
}

# Subroutine: printDevInfo
# Purpose: Print info about available devices
sub printDevInfo {
   my $mode     = $_[0];
   my $devNameString;
   my $retval;
   my $devNumString = "Number of devices";
   my $platNameString = "PlatformName";

   # If the user specified a host file, pass that to mpirun
   my $hostfileString;
   if ($hostfile eq "") {
    $hostfileString = "";
   } else {
    $hostfileString = "mpirun -np 1 -hostfile $hostfile ";
   }

   # Run a level 0 benchmark with the device info flag, and
   # figure out the number of available devices
   if (!$readonly) {
       if ($mode eq "cuda") {
          $devNameString = "name";
          my $command = $hostfileString . $bindir .
              "/Serial/CUDA/BusSpeedDownload -i > Logs/deviceInfo.txt 2> Logs/deviceInfo.err";
          open(DETECT,">Logs/detect.txt");
          print DETECT "Running this command to detect devices:\n$command\n";
          close(DETECT);
          $retval = system($command);
          # Collect info from nvidia utilities
          system("nvidia-smi -r -a >Logs/eccConfig.txt 2> Logs/eccConfig.err");
          system("nvcc --version >Logs/nvccVersion.txt 2> Logs/nvccVersion.err");
          system("cat /proc/driver/nvidia/version >Logs/driverVersion.txt 2>Logs/driverVersion.err");
       }
       else {
          $devNameString = "DeviceName";
          my $command = $hostfileString . $bindir .
              "/Serial/OpenCL/BusSpeedDownload -i > Logs/deviceInfo.txt 2> Logs/deviceInfo.err";
          open(DETECT,">Logs/detect.txt");
          print DETECT "Running this command to detect devices:\n$command\n";
          close(DETECT);
          $retval = system($command);
       }

       # Save the compiler version and build flags in the Logs
       # Assumes SHOC has been successfully installed.
       system("cp $bindir/../share/doc/shoc/buildFlags.txt Logs/buildFlags.txt");
       system("cp $bindir/../share/doc/shoc/compilerVersion.txt Logs/compilerVersion.txt");

       die "Error collecting device info.\n".
           "Make sure that you are running in the SHOC install root directory (or set -bindir)\n".
           "and any hostfile you set is correct.\n" unless $retval == 0;
   }

   # Now parse the device info, and figure out how many devices are available
   open( INFILE, "./Logs/deviceInfo.txt" );

   my $line;
   my @tokens;
   my @n;
   my @deviceNames;
   my @platformNames;
   my $platforms = 0;
   my $devCount = 0;

   while ( $line = <INFILE> ) {
       chomp($line);
       $line =~ s/^\s+//;    #remove leading spaces
       @tokens = split( /\s+=\s+/, $line );
       if ( $tokens[0] eq $devNumString ) {
           $devCount = 0; # Reset device count on every new plat
           $n[$platforms] = int( $tokens[1] ); # save count for plat
           $platforms = $platforms + 1; # this is a new plat so inc
       }
       if ( $tokens[0] eq $devNameString ) {
           $deviceNames[$platforms-1][$devCount] = $tokens[1];
           $devCount = $devCount + 1;
       }
       if ($tokens[0] eq $platNameString ) {
           $platformNames[$platforms] = $tokens[1];
       }
   }
   close(INFILE);
   print "Number of available platforms: $platforms \n";
   my $i;
   my $j;
   for ( $i = 0 ; $i < $platforms ; $i++ ) {
       print "Number of available devices on platform $i ", 
           $platformNames[$i] ,": ", $n[$i], "\n";
       for ( $j = 0 ; $j < $n[$i] ; $j++ ) {
           print "Device $j: ", $deviceNames[$i][$j], "\n";
       }
   }
   return;
}

# Subroutine: usage
# Purpose: Print command arguments and die
sub usage() {
   my $txt = $_[0];
   print "Usage: perl driver.pl [options]\n";
   print "Mandatory Options\n";
   print "-s      - Problem size (see SHOC wiki for specifics)\n";
   print "-cuda   - Use the cuda version of benchmarks\n";
   print "-opencl - Use the opencl version of the benchmarks\n";
   print "Note -cuda and -opencl are mutually exlcusive.\n\n";

   print "Other options\n";
   print "-n          - Number of nodes to run on\n";
   print "-p          - OpenCL Platform ID to run on\n";
   print "-d          - Comma-separated list of device numbers on each node\n";
   print "-hostfile   - specify hostfile for parallel runs\n";
   print "-help       - print this message\n";
   print "-bindir     - location of SHOC bin directory\n";
   print "-benchmark  - name of a single benchmark to run\n\n";
   print "Note: The driver script assumes it is running from the tools\n";
   print "directory.  Use -bindir when you need to run from somwhere else\n\n";

   print "Examples\n";
   print "Test all detected devices, using cuda and a large problem:\n";
   print "\$ perl driver.pl -cuda -s 4\n\n";
   print "Test a cluster with 4 nodes, first 3 devs on each node, using cuda and a hostfile\n";
   print "\$ perl driver.pl -n 4 -d 0,1,2 -cuda -hostfile hostfile_name\n";

   exit 0;
}

# Subroutine: checkError
# Purpose: Check to see if a benchmark returned an error
sub checkError() {
   my $ans = $_[0];
   # result DB reports FLT_MAX (guaranteed to be >= 1E+37?) for tests not run
   if ($ans eq "N/A" or $ans eq "NA") {
       return "NoResult";
   }
   # "0" and "inf" results are a sign of an error for some benchmarks
   # "-1" and "1e37" are internal sentinal values used in this script
   elsif ( $ans == 0 || $ans == -1 || $ans =~ /inf/ || $ans == 1E+37) {
      return "BenchmarkError";
   }
   else {
      return $ans;
   }
}


# Subroutine: buildFileName(testName, deviceNum)
# Purpose: Helper routine to construct fileNames
# testName -- name of test
# deviceNum -- device number
sub buildFileName {
    if (defined $_[1] and $_[1] ne "")
    {
        return "Logs/dev" . $_[1] . "_" . $_[0];
    }
    else
    {
        return "Logs/all_" . $_[0];
    }
}

