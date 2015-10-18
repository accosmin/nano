#pragma once

#include "sample.h"

namespace cortex
{
        class task_t;
        class loss_t;
        class model_t;

        ///
        /// \brief evaluate a model (compute the average loss value & error)
        ///
        NANOCV_PUBLIC size_t evaluate(const task_t& task, const fold_t& fold, const loss_t& loss, const model_t& model,
                scalar_t& lvalue, scalar_t& lerror);
}

