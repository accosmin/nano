#pragma once

#include "version.h"
#include "loss.h"
#include "layer.h"
#include "model.h"
#include "trainer.h"
#include "criterion.h"

namespace ncv
{
        ///
        /// \brief current version
        ///
        string_t version();

        ///
        /// \brief initialize library (setup flags, register default objects ...)
        ///
        void init();
}
