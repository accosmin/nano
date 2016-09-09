#pragma once

#include "arch.h"
#include <string>

#define NANO_MAJOR_VERSION            0
#define NANO_MINOR_VERSION            3
#define NANO_REVISION_VERSION         0

namespace nano
{
        ///
        /// \brief current version
        ///
        NANO_PUBLIC std::string version();
}
