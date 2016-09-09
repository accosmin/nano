#include "version.h"
#include "text/to_string.hpp"

std::string nano::version()
{
        return  nano::to_string(NANO_MAJOR_VERSION) + "." +
                nano::to_string(NANO_MINOR_VERSION) + "." +
                nano::to_string(NANO_REVISION_VERSION);
}


