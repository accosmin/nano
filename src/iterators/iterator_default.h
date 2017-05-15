#pragma once

#include "iterator.h"

namespace nano
{
        ///
        /// \brief use the samples as they are.
        ///
        struct iterator_default_t final : public iterator_t
        {
                explicit iterator_default_t(const string_t& configuration = string_t());

                virtual tensor3d_t input(const task_t&, const fold_t&, const size_t index) const final;
                virtual tensor3d_t target(const task_t&, const fold_t&, const size_t index) const final;
        };
}
