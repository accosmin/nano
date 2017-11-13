#pragma once

#include "factory.h"
#include "function.h"
#include "batch/params.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        class batch_solver_t;
        using batch_solver_factory_t = factory_t<batch_solver_t>;
        using rbatch_solver_t = batch_solver_factory_t::trobject;

        NANO_PUBLIC batch_solver_factory_t& get_batch_solvers();

        ///
        /// \brief generic batch solver that uses an adaptive line-search method.
        ///
        class NANO_PUBLIC batch_solver_t
        {
        public:

                virtual ~batch_solver_t() = default;

                ///
                /// \brief serialize the current parameters to json
                ///
                virtual void config(json_reader_t&) = 0;
                virtual void config(json_writer_t&) const = 0;

                ///
                /// \brief minimize starting from the initial point x0
                ///
                virtual function_state_t minimize(const batch_params_t&, const function_t&, const vector_t& x0) const = 0;
        };
}
