#pragma once

#include "sample.h"

namespace nano
{
        class task_t;
        class loss_t;
        class model_t;
        class criterion_t;

        ///
        /// \brief evaluate a model (compute the average loss value & error)
        ///
        NANO_PUBLIC size_t evaluate(const task_t&, const fold_t&, const loss_t&, const criterion_t&, const model_t&,
                                      scalar_t& lvalue, scalar_t& lerror);
}

