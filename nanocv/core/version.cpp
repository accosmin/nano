#include "version.h"
#include "text/to_string.hpp"

ncv::string_t ncv::version()
{
        return  text::to_string(NANOCV_MAJOR_VERSION) + "." +
                text::to_string(NANOCV_MINOR_VERSION) + "." +
                text::to_string(NANOCV_REVISION_VERSION);
}


