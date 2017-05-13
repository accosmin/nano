#pragma once

#include "iterator.h"

namespace nano
{
        ///
        /// \brief generate samples by adding noise to inputs.
        ///
        struct iterator_noise_t final : public iterator_t
        {
                explicit iterator_noise_t(const string_t& configuration = string_t());

                virtual tensor3d_t input(const task_t&, const fold_t&, const size_t index) const final;
                virtual tensor3d_t target(const task_t&, const fold_t&, const size_t index) const final;
        };
}
