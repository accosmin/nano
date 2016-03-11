#pragma once

#include "loss.h"
#include "layer.h"
#include "model.h"
#include "trainer.h"
#include "criterion.h"

namespace nano
{
        ///
        /// \brief initialize library (setup flags, register default objects ...)
        ///
        NANO_PUBLIC void init();
}
