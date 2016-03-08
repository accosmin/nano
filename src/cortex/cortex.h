#pragma once

#include "loss.h"
#include "layer.h"
#include "model.h"
#include "trainer.h"
#include "criterion.h"

namespace cortex
{
        ///
        /// \brief initialize library (setup flags, register default objects ...)
        ///
        ZOB_PUBLIC void init();
}
