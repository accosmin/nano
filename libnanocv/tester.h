#pragma once

#include "sample.h"

namespace ncv
{
        class task_t;
        class loss_t;
        class model_t;

        ///
        /// \brief evaluate a model (compute the average loss value & error)
        ///
        size_t test(const task_t& task, const fold_t& fold, const loss_t& loss, const model_t& model,
                scalar_t& lvalue, scalar_t& lerror);
}

