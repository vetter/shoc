#ifndef CTIMER_H
#define CTIMER_H

// ****************************************************************************
//  File:  CTimer.h
//
//  Purpose:
//    C versions to call the critical routines of the Timer class.
//
//  Programmer:  Jeremy Meredith
//  Creation:    October 22, 2007
//
// ****************************************************************************
#ifdef __cplusplus
extern "C" {
#endif
int    Timer_Start();
double Timer_Stop(int, const char *);
void   Timer_Insert(const char *, double);
#ifdef __cplusplus
}
#endif


#endif
