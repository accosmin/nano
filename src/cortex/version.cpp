#include "version.h"
#include "text/to_string.hpp"

cortex::string_t cortex::version()
{
        return  text::to_string(NANOCV_MAJOR_VERSION) + "." +
                text::to_string(NANOCV_MINOR_VERSION) + "." +
                text::to_string(NANOCV_REVISION_VERSION);
}


