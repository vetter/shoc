#!/usr/bin/perl

use strict;
my @elem_lengthsA = 0;
my @elem_lengthsB = 0;
my @lines;
my @lastLines;
my $before = 1;
my $after = 0;

while(<>){
    chomp();
    my $ln=$_;
    my @elems = split(/\t+/,$ln);
    if( $elems[0] eq "test"  && $elems[1] eq "atts" ){
        $before = 0;
    }
    if( $before == 0 && ($#elems == 0 || $elems[0] eq "Note:") ){
        $after = 1;
    }

    if( $before == 1 ){
        print "$ln\n";
    }elsif( $after == 1 ){
        push(@lastLines, $ln);
    }else{
        # push each line in an array so we retrieve it later
        push(@lines, $ln);
        my $i=0;
        # for each element in this line, find its length
        foreach(@elems){
            my $elem=$_;
            # ignore elements that are of zero length (split() splits *around* delimiters, so multiple consecutive delimiters are returned as zero length strings).
            if( length($elem) > 0 ){
                # keep track of the longest string per column
                my ($lenA, $lenB);
                if( $elem =~ /(\d*)\.(\d*)/ ){
                    $lenA = length($1);
                    $lenB = length($2);
                }else{
                    $lenA = 1;
                    $lenB = length($elem);
                }
                if($elem_lengthsA[$i] == 0 || $lenA > $elem_lengthsA[$i]){
                  $elem_lengthsA[$i] = $lenA;
                }
                if($elem_lengthsB[$i] == 0 || $lenB > $elem_lengthsB[$i]){
                  $elem_lengthsB[$i] = $lenB;
                }
                $i++
            }
        }
    }
}

# iterate over the input (that we've stored into the array @lines) and print it
foreach(@lines){
    my $ln = $_;
    my @elems = split(/\t+/,$ln);
    my $i=0;
    foreach(@elems){
        my $elem = $_;
        # skip delimiters
        if( length($elem) > 0 ){
            # find the maximum length of this column and use it as the string length (+1)
            my $tmp_len = 1+$elem_lengthsA[$i]+$elem_lengthsB[$i];
            if( $elem !~ /\d*\.\d*/ ){
                my $frmt = " %-".$tmp_len."s ";
                printf($frmt,$elem);
            }else {
                my $frmt = " %".$tmp_len.".".$elem_lengthsB[$i]."lf ";
                printf($frmt,$elem);
            }
            $i++;
        }
    }
    print "\n";
}

foreach(@lastLines){
    print "$_\n";
}
