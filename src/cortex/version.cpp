#include "version.h"
#include "text/to_string.hpp"

std::string cortex::version()
{
        return  text::to_string(CORTEX_MAJOR_VERSION) + "." +
                text::to_string(CORTEX_MINOR_VERSION) + "." +
                text::to_string(CORTEX_REVISION_VERSION);
}


