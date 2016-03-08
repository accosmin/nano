#pragma once

#include "arch.h"
#include <string>

#define CORTEX_MAJOR_VERSION            0
#define CORTEX_MINOR_VERSION            3
#define CORTEX_REVISION_VERSION         0

namespace cortex
{
        ///
        /// \brief current version
        ///
        ZOB_PUBLIC std::string version();
}
