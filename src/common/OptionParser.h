#ifndef OPTION_PARSER_H
#define OPTION_PARSER_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "Option.h"

using namespace std;

// ****************************************************************************
// Class:  OptionParser
//
// Purpose:
//   Class used to specify and parse command-line options to programs.
//
// Programmer:  Kyle Spafford
// Creation:    August 4, 2009
//
// ****************************************************************************
class OptionParser
{
  private:
    typedef std::map<std::string, Option> OptionMap;

    OptionMap optionMap;
    map<char, string>   shortLetterMap;

    bool helpRequested;

  public:

    OptionParser();
    void addOption(const string &longName,
                   OptionType type,
                   const string &defaultValue,
                   const string &helpText = "No help specified",
                   char shortLetter = '\0');

    void print() const;

    //Returns false on failure, true on success
    bool parse(int argc, const char *const argv[]);
    bool parse(const vector<string> &args);
    bool parseFile(const string &fileName);

    //Accessors for options
    long long   getOptionInt(const string &name) const;
    float       getOptionFloat(const string &name) const;
    bool        getOptionBool(const string &name) const;
    string      getOptionString(const string &name) const;

    vector<long long>     getOptionVecInt(const string &name) const;
    vector<float>         getOptionVecFloat(const string &name) const;
    vector<string>        getOptionVecString(const string &name) const;

    void printHelp(const string &optionName) const;
    void usage() const;

    bool HelpRequested( void ) const    { return helpRequested; }
};

#endif
