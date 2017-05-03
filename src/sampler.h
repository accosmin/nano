#pragma once

#include "arch.h"
#include "manager.h"

namespace nano
{
        ///
        /// \brief manage sampling objects (register new ones, query and clone them)
        ///
        struct sampler_t;
        using sampler_manager_t = manager_t<sampler_t>;
        using rsampler_t = task_manager_t::trobject;

        NANO_PUBLIC sampler_manager_t& get_samplers();

        ///
        /// \brief sample generator for augmenting the training dataset.
        ///
        struct NANO_PUBLIC sampler_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief transform the given {input tensor, its target and its label}
                ///
                virtual void get(tensor3d_t& input, vector_t& target, string_t& label) = 0;
        };
}
