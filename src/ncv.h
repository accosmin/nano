#ifndef NANOCV_H
#define NANOCV_H

#include "text.h"
#include "loss.h"
#include "layer.h"
#include "task.h"
#include "model.h"
#include "trainer.h"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"
#include "util/logger.h"
#include "util/timer.h"
#include "util/math.hpp"

namespace ncv
{
        // current version
        static const size_t MAJOR_VERSION = 0;
        static const size_t MINOR_VERSION = 1;

        inline string_t version()
        {
                return text::to_string(MAJOR_VERSION) + "." +
                       text::to_string(MINOR_VERSION);
        }

        // initialize library (register objects, start worker pool ...)
        void init();

        // evaluate a model (compute the average loss value & error)
        size_t test(const task_t& task, const fold_t& fold, const loss_t& loss, const model_t& model,
                scalar_t& lvalue, scalar_t& lerror);
}

#endif // NANOCV_H
