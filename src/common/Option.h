#ifndef OPTION_H
#define OPTION_H

#include <string>

using namespace std;

enum OptionType {OPT_FLOAT, OPT_INT, OPT_STRING, OPT_BOOL,
                 OPT_VECFLOAT, OPT_VECINT, OPT_VECSTRING};

// ****************************************************************************
// Class:  Option
//
// Purpose:
//   Encapsulation of a single option, to be used by an option parser.
//
// Programmer:  Kyle Spafford
// Creation:    August 4, 2009
//
// ****************************************************************************
class Option {

  public:

   string longName;
   char   shortLetter;
   string defaultValue;
   string value;
   OptionType type;
   string helpText;

   void print();
};

#endif
