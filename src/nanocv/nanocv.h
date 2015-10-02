#pragma once

#include "cortex/loss.h"
#include "cortex/layer.h"
#include "cortex/model.h"
#include "cortex/trainer.h"
#include "cortex/criterion.h"

namespace ncv
{
        ///
        /// \brief initialize library (setup flags, register default objects ...)
        ///
        NANOCV_PUBLIC void init();
}
