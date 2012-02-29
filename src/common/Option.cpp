#include <iostream>
#include "Option.h"
using namespace std;

void Option::print() {

   cout << "Long Name: " << longName << endl;
   cout << "Short Name: " << shortLetter << endl;
   cout << "Default Value: " << defaultValue << endl;
   cout << "Actual Value: " << value << endl;
   cout << "Type: " << type << endl;
   cout << "helpText: " << helpText << endl;

}
