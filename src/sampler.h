#pragma once

#include "task.h"

namespace nano
{
        ///
        /// \brief manage sampling objects (register new ones, query and clone them)
        ///
        struct sampler_t;
        using sampler_manager_t = manager_t<sampler_t>;
        using rsampler_t = sampler_manager_t::trobject;

        NANO_PUBLIC sampler_manager_t& get_samplers();

        ///
        /// \brief sample generator for augmenting the training dataset.
        ///
        struct NANO_PUBLIC sampler_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief retrieve the 3D input tensor for a given sample
                ///
                virtual tensor3d_t input(const task_t&, const fold_t&, const size_t index) = 0;

                ///
                /// \brief retrieve the output target for a given sample
                ///
                virtual tensor3d_t target(const task_t&, const fold_t&, const size_t index) = 0;
        };
}
