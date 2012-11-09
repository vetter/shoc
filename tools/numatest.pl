#! /usr/bin/env perl

# Parse the arguments
$platform = "OpenCL";
while ($_ = shift @ARGV)
{
    if (/^-cuda$/ or /^--cuda$/)
    {
	$platform = "CUDA";
    }
    elsif (/^-opencl$/ or /^--opencl$/)
    {
	$platform = "OpenCL";
    }
    else
    {
	print STDERR "Unknown argument: '$_'\n";
	print STDERR "\n";
	print STDERR "Usage: $0 [--cuda | --opencl]\n";
	print STDERR "       (defaults to OpenCL)\n";
	print STDERR "\n";
	exit 1;
    }
}
print "Using platform: $platform\n";

# Get the CUDA/OpenCL devices available
@devicequeryoutput = `../bin/Serial/$platform/BusSpeedDownload -i`;
$num_devs = (grep(/Number of devices/, @devicequeryoutput))[0];
$num_devs =~ s/^.*=\s*//;
chomp($num_devs);
print "Number of $platform devices: $num_devs\n";

# Get the NUMA nodes available
@numaoutput = `numactl --show`;
$numa_node_str = (grep(/nodebind/, @numaoutput))[0];
$numa_node_str =~ s/^.*:\s*//;
chomp($numa_node_str);
@numa_nodes = split /\s+/, $numa_node_str;
print "Number of NUMA nodes= @numa_nodes\n";

# Check download speed and latency for all NUMA node / device pairings
foreach $n (@numa_nodes)
{
    for ($d = 0; $d < $num_devs; $d++)
    {
	@down_output = `numactl --cpunodebind=$n ../bin/Serial/$platform/BusSpeedDownload -d $d`;

	$bw_str = (grep(/DownloadSpeed\s+65536kB/, @down_output))[0];
	@bw_cols = split /\s+/, $bw_str;
	$bw_median = $bw_cols[3];

	$lat_str = (grep(/DownloadTime\s+1kB/, @down_output))[0];
	@lat_cols = split /\s+/, $lat_str;
	$lat_median = $lat_cols[3];

	print "NUMA Node=$n Device=$d Median Download Latency=$lat_median ms, Speed=$bw_median GB/sec\n";
    }
}

# Check readback speed and latency for all NUMA node / device pairings
foreach $n (@numa_nodes)
{
    for ($d = 0; $d < $num_devs; $d++)
    {
	@up_output = `numactl --cpunodebind=$n ../bin/Serial/$platform/BusSpeedReadback -d $d`;

	$bw_str = (grep(/ReadbackSpeed\s+65536kB/, @up_output))[0];
	@bw_cols = split /\s+/, $bw_str;
	$bw_median = $bw_cols[3];

	$lat_str = (grep(/ReadbackTime\s+1kB/, @up_output))[0];
	@lat_cols = split /\s+/, $lat_str;
	$lat_median = $lat_cols[3];

	print "NUMA Node=$n Device=$d Median Upload Latency=$lat_median ms, Speed=$bw_median GB/sec\n";
    }    
}
