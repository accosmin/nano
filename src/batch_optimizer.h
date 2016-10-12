#pragma once

#include "problem.h"
#include "manager.h"
#include "batch/params.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        class batch_optimizer_t;
        using batch_optimizer_manager_t = manager_t<batch_optimizer_t>;

        NANO_PUBLIC batch_optimizer_manager_t& get_batch_optimizers();

        ///
        /// \brief generic batchastic optimizer
        ///
        class NANO_PUBLIC batch_optimizer_t : public clonable_t<batch_optimizer_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit batch_optimizer_t(const string_t& configuration = string_t()) :
                        clonable_t<batch_optimizer_t>(configuration)
                {
                }

                ///
                /// \brief minimize starting from the initial point x0.
                ///
                virtual state_t minimize(const batch_params_t&, const problem_t&, const vector_t& x0) const = 0;
        };
}
