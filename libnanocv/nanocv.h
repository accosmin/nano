#pragma once

#include "loss.h"
#include "layer.h"
#include "model.h"
#include "trainer.h"
#include "version.h"
#include "criterion.h"
#include "util/arch.h"

namespace ncv
{
        ///
        /// \brief current version
        ///
        NANOCV_DLL_PUBLIC string_t version();

        ///
        /// \brief initialize library (setup flags, register default objects ...)
        ///
        NANOCV_DLL_PUBLIC void init();
}
