#pragma once

#include "arch.h"
#include "string.h"

#define NANOCV_MAJOR_VERSION            0
#define NANOCV_MINOR_VERSION            2
#define NANOCV_REVISION_VERSION         0

namespace cortex
{
        ///
        /// \brief current version
        ///
        NANOCV_PUBLIC string_t version();
}
