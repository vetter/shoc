#ifndef _PROGRESS_BAR_H_
#define _PROGRESS_BAR_H_

#include <stdio.h>
#include <stdlib.h>

#ifndef _WIN32
#include <unistd.h>
#endif


// ****************************************************************************
// Class: ProgressBar
//
// Purpose:
//   Simple text progress bar class.
//
// Programmer: Gabriel Marin
// Creation:   October 12, 2009
//
// Modifications:
//
// ****************************************************************************
class ProgressBar
{
private:
    int itersDone;
    int totalIters;
    static const char barDone[81];
    double rTotal;
    double percDone;

public:
    //   Constructor
    //
    //   Arguments:
    //       _totalIters  total work amount to be tracked
    ProgressBar (int _totalIters = 0)
    {
        totalIters = _totalIters;
        itersDone = 0;
        if (totalIters)
        {
            rTotal = 100.0/totalIters;
        } else
        {
            rTotal = 0.0;
        }
        percDone = itersDone*rTotal;
    }

    //   Method: setTotalIters
    //
    //   Purpose: setter for the total work amount
    //
    //   Arguments:
    //       _totalIters  total work amount to be tracked
    void setTotalIters (int _totalIters)
    {
        totalIters = _totalIters;
        if (totalIters)
        {
            rTotal = 100.0/totalIters;
            percDone = itersDone*rTotal;
        }
    }

    //   Method: setItersDone
    //
    //   Purpose: setter for the completed work amount
    //
    //   Arguments:
    //       _itersDone  completed work amount
    void setItersDone (int _itersDone)
    {
        itersDone = _itersDone;
        percDone = itersDone*rTotal;
    }

    //   Method: addItersDone
    //
    //   Purpose: update amount of completed work
    //
    //   Arguments:
    //       _inc  amount of newly completed work
    void addItersDone (int _inc = 1)
    {
        itersDone += _inc;
        percDone = itersDone*rTotal;
    }

    //   Method: Show
    //
    //   Purpose: display progress bar
    //
    //   Arguments:
    //       fd  output file descriptor
    void Show (FILE *fd)
    {
        int lenDone = (int)(percDone/2.0 + 0.5);
        fprintf(fd, "\r|%.*s%*s| %5.1lf%%", lenDone, barDone, 50-lenDone, "", percDone);
        fflush(fd);
    }
};

#endif
