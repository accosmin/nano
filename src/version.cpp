#include "version.h"
#include "text/to_string.hpp"

std::string nano::version()
{
        return  nano::to_string(CORTEX_MAJOR_VERSION) + "." +
                nano::to_string(CORTEX_MINOR_VERSION) + "." +
                nano::to_string(CORTEX_REVISION_VERSION);
}


