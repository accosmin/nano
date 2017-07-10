#pragma once

#include "factory.h"
#include "trainer_result.h"

namespace nano
{
        struct loss_t;
        struct task_t;
        struct model_t;
        struct iterator_t;
        struct accumulator_t;

        ///
        /// \brief stores registered prototypes
        ///
        struct trainer_t;
        using trainer_factory_t = factory_t<trainer_t>;
        using rtrainer_t = trainer_factory_t::trobject;

        NANO_PUBLIC trainer_factory_t& get_trainers();

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
                        const iterator_t&, const task_t&, const size_t fold, accumulator_t&) const = 0;
        };
}
