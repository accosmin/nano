#pragma once

#include "manager.h"
#include "trainer_result.h"

namespace nano
{
        struct loss_t;
        struct task_t;
        struct model_t;
        class sampler_t;
        class criterion_t;
        class accumulator_t;

        ///
        /// \brief stores registered prototypes
        ///
        struct trainer_t;
        using trainer_manager_t = manager_t<trainer_t>;
        using rtrainer_t = trainer_manager_t::trobject;

        NANO_PUBLIC trainer_manager_t& get_trainers();

        ///
        /// \brief generic trainer: optimizes a model on a given compatible task.
        ///
        struct NANO_PUBLIC trainer_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief train the given model starting from the current model parameters
                ///
                virtual trainer_result_t train(
                        const task_t&, const size_t fold, const size_t nthreads,
                        const loss_t&, const criterion_t& criterion,
                        model_t&) const = 0;
        };
}

