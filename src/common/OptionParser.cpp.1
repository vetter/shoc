#include <string>
#include <vector>
#include <map>
#include "OptionParser.h"
#include "Utility.h"
#include <sstream>
#include <fstream>
#include <iomanip>

using namespace std;

OptionParser::OptionParser()
  : helpRequested( false )
{
    addOption("configFile", OPT_STRING, "",
              "specify configuration file", 'c');
    addOption("help", OPT_BOOL, "", "print this usage", 'h');
}

void OptionParser::addOption(const string &longName,
                             OptionType type,
                             const string &defaultValue,
                             const string &helpText,
                             char shortLetter)
{
  Option opt;
  opt.longName = longName;
  opt.type = type;
  opt.defaultValue = defaultValue;
  opt.value = defaultValue;
  opt.helpText = helpText;
  opt.shortLetter = shortLetter;
  if (optionMap.count(longName)>0)
      cout << "Internal error: used option long name '"<<longName<<"' twice.\n";
  optionMap[longName] = opt;
  if (shortLetter != '\0')
  {
      if (shortLetterMap.count(shortLetter)>0)
          cout << "Internal error: used option short letter '"
               << shortLetter<<"' twice (for '"
               << shortLetterMap[opt.shortLetter]
               << "' and '"
               << longName <<"')\n";
      shortLetterMap[opt.shortLetter] = opt.longName;
  }
}

bool OptionParser::parse(int argc, const char *const argv[])
{
    vector<string> args;
    for (int i=1; i<argc; i++)
        args.push_back(argv[i]);
    return parse(args);
}

//
// Modifications:
//   Jeremy Meredith, Thu Nov  4 14:42:18 EDT 2010
//   Don't print out usage here; count on the caller to do that.
//   The main reason is we parse the options in parallel and
//   don't want every task to print an error or help text.
//
bool OptionParser::parse(const vector<string> &args) {

   for (int i=0; i<args.size(); i++) {

      //parse arguments
      string temp = args[i];
      if (temp[0] != '-')
      {
         cout << "failure, no leading - in option: " << temp << "\n";
         cout << "Ignoring remaining options" << endl;
         return false;
      }
      else if (temp[0] == '-' && temp[1] == '-') //Long Name argument
      {
         string longName = temp.substr(2);
         if (longName=="configFile" && ! (i+1>=args.size())) {
             if (!parseFile(args[i+1]))
             {
                 return false;
             }
             i++;
             continue;
         }
         if (optionMap.find(longName) == optionMap.end()) {
            cout << "Option not recognized: " << temp << endl;
            cout << "Ignoring remaining options" << endl;
            return false;
         }
         if (optionMap[longName].type == OPT_BOOL) {
            //Option is bool and is flagged true
            optionMap[longName].value = "true";
         } else {
            if (i+1 >= args.size()) {
               cout << "failure, option: " << temp << " with no value\n";
               cout << "Ignoring remaining options" << endl;
               return false;
            } else {
               optionMap[longName].value = args[i+1];
               i++;
            }
         }
      }
      else  //Short name argument
      {
          int nopts = temp.length()-1;
          for (int p=0; p<nopts; p++)
          {
              char shortLetter = temp[p+1];
              if (shortLetterMap.find(shortLetter) == shortLetterMap.end()) {
                  cout << "Option: " << temp << " not recognized.\n";
                  cout << "Ignoring remaining options" << endl;
                  return false;
              }
              string longName = shortLetterMap[shortLetter];
              if (longName=="configFile" && ! (i+1>=args.size())) {
                  if (!parseFile(args[i+1]))
                  {
                      return false;
                  }
                  i++;
                  continue;
              }

              if (optionMap[longName].type == OPT_BOOL) {
                  //Option is bool and is flagged true
                  optionMap[longName].value = "true";
              } else {
                  if (i+1 >= args.size() || p < nopts-1)
                  {
                      //usage();
                      cout << "failure, option: -" << shortLetter << " with no value\n";
                      cout << "Ignoring remaining options" << endl;
                      return false;
                  }
                  else
                  {
                      optionMap[longName].value = args[i+1];
                      i++;
                  }
              }
          }
      }
   }

   if (getOptionBool("help"))
   {
       helpRequested = true;
       return false;
   }

   return true;
}

void OptionParser::print() const {

   vector<string> printed;

    OptionMap::const_iterator i = optionMap.begin();
   cout << "Printing Options" << endl;
   while(i != optionMap.end()) {
      Option o = i->second;
      bool skip = false;
      for (int j=0; j<printed.size(); j++) {
         if (printed[j] == o.longName) skip = true;
      }
      if (!skip) {
        printed.push_back(o.longName);
        o.print();
        i++;
        cout << "---------------------" << endl;
      } else { i++; }
   }

}

long long OptionParser::getOptionInt(const string &name) const {

   long long retVal;

    OptionMap::const_iterator iter = optionMap.find( name );
   if (iter == optionMap.end()) {
     cout << "getOptionInt: option name \"" << name << "\" not recognized.\n";
     return -9999;
   }

   stringstream ss(iter->second.value);
   ss >> retVal;

   return retVal;

}

float OptionParser::getOptionFloat(const string &name) const {

   float retVal;

    OptionMap::const_iterator iter = optionMap.find( name );
   if (iter == optionMap.end()) {
     cout << "getOptionFloat: option name \"" << name << "\" not recognized.\n";
     return -9999;
   }

   stringstream ss(iter->second.value);
   ss >> retVal;

   return retVal;
}

bool OptionParser::getOptionBool(const string &name) const {

   int retVal;

    OptionMap::const_iterator iter = optionMap.find( name );
   if (iter == optionMap.end()) {
     cout << "getOptionBool: option name \"" << name << "\" not recognized.\n";
     return false;
   }

   return (iter->second.value == "true");
}
string OptionParser::getOptionString(const string &name) const {
    OptionMap::const_iterator iter = optionMap.find( name );
   if (iter == optionMap.end()) {
     cout << "getOptionString: option name \"" << name << "\" not recognized.\n";
     return "ERROR - Option not recognized";
   }
   return iter->second.value;
}

vector<long long> OptionParser::getOptionVecInt(const string &name) const {


   vector<long long> retval = vector<long long>(0);
    OptionMap::const_iterator iter = optionMap.find( name );
   if (iter == optionMap.end()) {
     cout << "getOptionVecInt: option name \"" << name << "\" not recognized.\n";
     return retval;
   }

   vector<string> tokens = SplitValues(iter->second.value, ',');
   for (int i=0; i<tokens.size(); i++) {
       stringstream ss(tokens[i]);
       long long j;
       ss >> j;
       retval.push_back(j);
   }
   return retval;
}
vector<float> OptionParser::getOptionVecFloat(const string &name) const {

   vector<float> retval = vector<float>(0);
    OptionMap::const_iterator iter = optionMap.find( name );
   if (iter == optionMap.end()) {
     cout << "getOptionVecFloat: option name \"" << name << "\" not recognized.\n";
     return retval;
   }
   vector<string> tokens = SplitValues(iter->second.value, ',');
   for (int i=0; i<tokens.size(); i++) {
       stringstream ss(tokens[i]);
       float f;
       ss >> f;
       cout << "F: " << f << endl;
       retval.push_back(f);
   }
   return retval;
}

vector<string> OptionParser::getOptionVecString(const string &name) const {

   vector<string> retval = vector<string>(0);
    OptionMap::const_iterator iter = optionMap.find( name );
   if (iter == optionMap.end()) {
     cout << "getOptionVecString: option name \"" << name << "\" not recognized.\n";
     return retval;
   }
   vector<string> tokens = SplitValues(iter->second.value, ',');
   for (int i=0; i<tokens.size(); i++) {
       stringstream ss(tokens[i]);
       string s;
       ss >> s;
       retval.push_back(s);
   }
   return retval;
}


void OptionParser::printHelp(const string &optionName) const {

    OptionMap::const_iterator iter = optionMap.find( optionName );
   if (iter == optionMap.end()) {
     cout << "printHelp: option name \"" << optionName << "\" not recognized.\n";
   } else {
      cout << iter->second.helpText;
   }
}

bool OptionParser::parseFile(const string &fileName) {

   ifstream inf(fileName.c_str());
   string line;
   vector<string> optionsFromFile;

   if (!inf.good()) {
      cout << "Bad config file" << endl;
      return false;
   }

   while (!getline(inf, line).eof())
   {
      if (line[0] == '#')
         continue;
      else {
         vector<string> tokens = SplitValues(line, ' ');
         for (int i=0; i<tokens.size(); i++) {
            if (i==0) {
                optionsFromFile.push_back("--"+tokens[i]);
             } else {
                optionsFromFile.push_back(tokens[i]);
             }
         }
      }
   }

   inf.close();
   return parse(optionsFromFile);
}


void OptionParser::usage() const {


   string type;
   cout << "Usage: benchmark ";
    OptionMap::const_iterator j = optionMap.begin();

   Option jo = j->second;
   cout << "[--" << jo.longName << " ";

   if (jo.type == OPT_INT || jo.type == OPT_FLOAT)
      type = "number";
   else if (jo.type == OPT_BOOL)
      type = "";
   else if (jo.type == OPT_STRING)
      type = "value";
   else if (jo.type == OPT_VECFLOAT || jo.type == OPT_VECINT)
      type = "n1,n2,...";
   else if (jo.type == OPT_VECSTRING)
      type = "value1,value2,...";
   cout << type << "]" << endl;

   while (++j !=optionMap.end()) {
      jo = j->second;
      cout << "                 [--" << jo.longName << " ";

      if (jo.type == OPT_INT || jo.type == OPT_FLOAT)
          type = "number";
      else if (jo.type == OPT_BOOL)
          type = "";
      else if (jo.type == OPT_STRING)
         type = "value";
      else if (jo.type == OPT_VECFLOAT || jo.type == OPT_VECINT)
          type = "n1,n2,...";
      else if (jo.type == OPT_VECSTRING)
          type = "value1,value2,...";

      cout << type << "]" << endl;
   }

   cout << endl;
   cout << "Available Options: " << endl;
    OptionMap::const_iterator i = optionMap.begin();
   while(i != optionMap.end()) {
      Option o = i->second;
      cout << "    ";
      if (o.shortLetter)
          cout << "-" << o.shortLetter << ", ";
      else
          cout << "    ";
      cout << setiosflags(ios::left) << setw(25)
           << "--" + o.longName + "    " << o.helpText << endl;

      i++;
   }

}

