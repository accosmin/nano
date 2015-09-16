#pragma once

#include "ml/loss.h"
#include "ml/layer.h"
#include "ml/model.h"
#include "ml/trainer.h"
#include "ml/criterion.h"

namespace ncv
{
        ///
        /// \brief current version
        ///
        NANOCV_PUBLIC string_t version();

        ///
        /// \brief initialize library (setup flags, register default objects ...)
        ///
        NANOCV_PUBLIC void init();
}
