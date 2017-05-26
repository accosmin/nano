#pragma once

#include "factory.h"
#include "function.h"
#include "batch/params.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        struct batch_optimizer_t;
        using batch_optimizer_factory_t = factory_t<batch_optimizer_t>;
        using rbatch_optimizer_t = batch_optimizer_factory_t::trobject;

        NANO_PUBLIC batch_optimizer_factory_t& get_batch_optimizers();

        ///
        /// \brief generic batchastic optimizer
        ///
        struct NANO_PUBLIC batch_optimizer_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief minimize starting from the initial point x0
                ///
                virtual function_state_t minimize(const batch_params_t&, const function_t&, const vector_t& x0) const = 0;
        };
}
