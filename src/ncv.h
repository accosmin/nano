#ifndef NANOCV_H
#define NANOCV_H

#include "core/text.h"
#include "core/random.hpp"
#include "core/logger.h"
#include "core/timer.h"
#include "loss.h"
#include "layer.h"
#include "task.h"
#include "model.h"
#include "trainer.h"

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
