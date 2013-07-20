#ifndef NANOCV_H
#define NANOCV_H

#include "core/optimize.h"
#include "core/geom.h"
#include "core/logger.h"
#include "core/random.h"
#include "core/timer.h"
#include "core/thread.h"
#include "core/image.h"
#include "core/stats.h"
#include "loss/loss.h"
#include "activation/activation.h"
#include "task/task.h"
#include "model/model.h"
#include "model/trainer.h"

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
        size_t test(const model_t& model, const task_t& task, const fold_t& fold, const loss_t& loss,
                scalar_t& lvalue, scalar_t& lerror);
}

#endif // NANOCV_H
