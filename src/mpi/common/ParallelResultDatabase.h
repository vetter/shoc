#ifndef PARALLEL_RESULT_DATABASE_H
#define PARALLEL_RESULT_DATABASE_H

#include "ParallelHelpers.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

// ****************************************************************************
// Class:  ParallelResultDatabase
//
// Purpose:
//   Unifies ResultDatabases over multiple processors.
//
// Programmer:  Jeremy Meredith
// Creation:    August 14, 2009
//
// Modifications:
//   Jeremy Meredith, Tue Jan 12 14:39:40 EST 2010
//   Switched to new ParAllGather.  A simple gather was giving horrendous
//   results on one machine, but allgather avoided the problem.
//
//   Jeremy Meredith, Wed Nov 10 14:25:42 EST 2010
//   Added DumpOutliers, which reports values from the per-process means
//   that fall more than <x> standard deviations from the all-processor mean.
//
//   Jeremy Meredith, Thu Nov 11 11:42:07 EST 2010
//   In MergeSerialDatabases, detect processors tagged as having missing
//   values (e.g. when running with some cards that do and some that don't
//   support double precision), remove the bad results, and flag them
//   in the test result name so they can be noted.
//
//   Jeremy Meredith, Mon Nov 22 13:36:03 EST 2010
//   Changed to use a quartile-based approach to outliers, as it should be
//   more robust.  Also, don't print results from a less-stringent
//   outlier test if they're going to pass a more-stringent outlier test.
//
//   Jeremy Meredith, Fri Dec  3 16:30:31 EST 2010
//   Removed "GetMean"; added a new function "GetResultsForTest" to base
//   class instead which is a little more general.  Use new GetResults
//   method as well to avoid a friend declaration.
//
// ****************************************************************************
static bool RemoveFLTMAXValues(vector<double> &a)
{
    int n = a.size();
    vector<double> b;
    b.reserve(n);
    for (int i=0; i<n; ++i)
    {
        if (a[i] < FLT_MAX)
            b.push_back(a[i]);
    }
    a.swap(b);
    return (a.size() < n);
}

class ParallelResultDatabase : public ResultDatabase
{
  public:
    void MergeSerialDatabases(ResultDatabase &db, MPI_Comm comm)
    {
        results.clear();
        int numResults = db.GetResults().size();
        MPI_Barrier(comm);
        for (int i = 0; i < numResults; i++)
        {
            const Result &r = db.GetResults()[i];

            // We tag any missing values (e.g. from non-double-precision
            // supporting cards) with FLT_MAX.
            bool hadMissingValues = r.HadAnyFLTMAXValues();

            // Propagate missing value tag to what gets collected in parallel.
            double mymin    = hadMissingValues ? FLT_MAX : r.GetMin();
            double mymax    = hadMissingValues ? FLT_MAX : r.GetMax();
            double mymedian = hadMissingValues ? FLT_MAX : r.GetMedian();
            double mymean   = hadMissingValues ? FLT_MAX : r.GetMean();
            double mystddev = hadMissingValues ? FLT_MAX : r.GetStdDev();

            // Gather the values.
            vector<double> allmin    = ParAllGather(mymin, comm);
            vector<double> allmax    = ParAllGather(mymax, comm);
            vector<double> allmedian = ParAllGather(mymedian, comm);
            vector<double> allmean   = ParAllGather(mymean, comm);
            vector<double> allstddev = ParAllGather(mystddev, comm);

            // Remove any missing value tags.
            bool foundTaggedValues = false;
            foundTaggedValues |= RemoveFLTMAXValues(allmin);
            foundTaggedValues |= RemoveFLTMAXValues(allmax);
            foundTaggedValues |= RemoveFLTMAXValues(allmedian);
            foundTaggedValues |= RemoveFLTMAXValues(allmean);
            foundTaggedValues |= RemoveFLTMAXValues(allstddev);

            // If we found any, put an asterisk in our report.
            string tag = foundTaggedValues ? "(*)" : "";
            AddResults(r.test + "(min)"   +tag, r.atts, r.unit, allmin);
            AddResults(r.test + "(max)"   +tag, r.atts, r.unit, allmax);
            AddResults(r.test + "(median)"+tag, r.atts, r.unit, allmedian);
            AddResults(r.test + "(mean)"  +tag, r.atts, r.unit, allmean);
            AddResults(r.test + "(stddev)"+tag, r.atts, r.unit, allstddev);
        }
    }
    void DumpOutliers(ostream &out)
    {
        // get only the mean results
        vector<Result> means;
        for (int i=0; i<results.size(); i++)
        {
            Result &r = results[i];
            if (r.test.length() > 6 &&
                r.test.substr(r.test.length()-6) == "(mean)")
            {
                means.push_back(r);
            }
        }

        // sort them
        sort(means.begin(), means.end());

        // get the max trials (in this case processors)
        int maxtrials = 1;
        for (int i=0; i<means.size(); i++)
        {
            if (means[i].value.size() > maxtrials)
                maxtrials = means[i].value.size();
        }

        out << "\nDetecting outliers based on per-process mean values.\n";

        // List of IQR thresholds to test.  Please put these in
        // increasing order so we can avoid reporting outliers twice.
        int nOutlierThresholds = 2;
        const char *outlierHeaders[] = {
            "Mild outliers (>1.5 IQR from 1st/3rd quartile)",
            "Extreme outliers (>3.0 IQR from 1st/3rd quartile)"
        };
        double outlierThresholds[] = {
            1.5,
            3.0
        };

        // for each threshold category, print any values
        // which are more than that many stddevs from the
        // all-processor-mean
        for (int pass=0; pass < nOutlierThresholds; pass++)
        {
            out << "\n" << outlierHeaders[pass]<< ":\n";
            bool foundAny = false;

            for (int i=0; i<means.size(); i++)
            {
                Result &r = means[i];
                double allProcsQ1 = r.GetPercentile(25);
                double allProcsQ3 = r.GetPercentile(75);
                double allProcsIQR = allProcsQ3 - allProcsQ1;
                double thresholdIQR = outlierThresholds[pass];
                double nextThresholdIQR = (pass < nOutlierThresholds-1) ?
                                                 outlierThresholds[pass+1] : 0;
                for (int j=0; j<r.value.size(); j++)
                {
                    double v = r.value[j];
                    if (v < allProcsQ1 - thresholdIQR * allProcsIQR)
                    {
                        // if they pass the next, more strict threshold,
                        // don't report it in this pass
                        if (pass == nOutlierThresholds-1 ||
                            v >= allProcsQ1 - nextThresholdIQR * allProcsIQR)
                        {
                            foundAny = true;
                            out << r.test << " " << r.atts << " "
                                << "Processor "<<j<<" had a mean value of "
                                << v << " " << r.unit <<", which is less than "
                                << "Q1 - IQR*" << thresholdIQR << ".\n";
                        }
                    }
                    else if (v > allProcsQ3 + thresholdIQR * allProcsIQR)
                    {
                        // if they pass the next, more strict threshold,
                        // don't report it in this pass
                        if (pass == nOutlierThresholds-1 ||
                            v <= allProcsQ3 + nextThresholdIQR * allProcsIQR)
                        {
                            foundAny = true;
                            out << r.test << " " << r.atts << " "
                                << "Processor "<<j<<" had a mean value of "
                                << v << " " << r.unit <<", which is more than "
                                << "Q3 + IQR*" << thresholdIQR << ".\n";
                        }
                    }
                }
            }
            // If we didn't find any, let them know that explicitly.
            if (!foundAny)
            {
                out << "None.\n";
            }
        }
    }

};

#endif
