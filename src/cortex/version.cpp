#include "version.h"
#include "text/to_string.hpp"

std::string zob::version()
{
        return  zob::to_string(CORTEX_MAJOR_VERSION) + "." +
                zob::to_string(CORTEX_MINOR_VERSION) + "." +
                zob::to_string(CORTEX_REVISION_VERSION);
}


