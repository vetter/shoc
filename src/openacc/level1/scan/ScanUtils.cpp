
// These dummary functions are needed by the TP version to 
// keep the compiler from thinking that a variable is dead, so that
// it will copy it from the device when we need it.
// Although it is unused for the Serial and EP versions, we have to
// define it because all of the TP, EP, and Serial versions share
// the same OpenACC benchmark function and that function refers to the dummy.
extern "C"
void
DummyFloatFunc( float /* v */ )
{
    // nothing else to do
}

extern "C"
void
DummyDoubleFunc( double /* v */ )
{
    // nothing else to do
}

