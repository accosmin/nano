#pragma once

#include "version.h"
#include "loss.h"
#include "layer.h"
#include "trainer.h"
#include "sampler.h"
#include "accumulator.h"
#include "util/measure.hpp"
#include "util/math.hpp"
#include "util/stats.hpp"
#include "util/thread_loop.hpp"
#include "util/random.hpp"

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
