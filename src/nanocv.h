#pragma once

#include "version.h"
#include "loss.h"
#include "layer.h"
#include "trainer.h"
#include "sampler.h"
#include "accumulator.h"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"
#include "optimize/opt_stoch_sg.hpp"
#include "optimize/opt_stoch_sga.hpp"
#include "optimize/opt_stoch_sia.hpp"
#include "optimize/opt_stoch_nag.hpp"
#include "common/measure.hpp"
#include "common/math.hpp"
#include "common/stats.hpp"
#include "common/thread_loop.hpp"
#include "common/random.hpp"

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

        ///
        /// \brief evaluate a model (compute the average loss value & error)
        ///
        size_t test(const task_t& task, const fold_t& fold, const loss_t& loss, const model_t& model,
                scalar_t& lvalue, scalar_t& lerror);
}
