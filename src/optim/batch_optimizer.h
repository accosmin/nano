#pragma once

#include "state.h"
#include "problem.h"
#include "manager.hpp"
#include "batch/params.h"

namespace nano
{
        class batch_optimizer_t;

        ///
        /// \brief stores registered prototypes
        ///
        using batch_optimizer_manager_t = manager_t<batch_optimizer_t>;
        using rbatch_optimizer_t = batch_optimizer_manager_t::trobject;

        NANO_PUBLIC batch_optimizer_manager_t& get_batch_optimizers();

        ///
        /// \brief generic batchastic optimizer
        ///
        class NANO_PUBLIC batch_optimizer_t : public clonable_t<batch_optimizer_t>
        {
        public:

                using clonable_t<batch_optimizer_t>::get_param;

                ///
                /// \brief constructor
                ///
                batch_optimizer_t(const string_t& configuration = string_t()) :
                        clonable_t<batch_optimizer_t>(configuration)
                {
                }

                ///
                /// \brief minimize starting from the initial point x0.
                ///
                virtual state_t minimize(const batch_params_t&, const problem_t&, const vector_t& x0) const = 0;
        };
}
