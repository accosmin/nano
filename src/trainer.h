#pragma once

#include "factory.h"
#include "trainer_result.h"

namespace nano
{
        class loss_t;
        struct task_t;
        class model_t;
        struct enhancer_t;
        struct accumulator_t;

        ///
        /// \brief stores registered prototypes
        ///
        class trainer_t;
        using trainer_factory_t = factory_t<trainer_t>;
        using rtrainer_t = trainer_factory_t::trobject;

        NANO_PUBLIC trainer_factory_t& get_trainers();

        ///
        /// \brief generic trainer: optimizes a model on a given compatible task.
        ///
        class NANO_PUBLIC trainer_t : public configurable_t
        {
        public:
                using configurable_t::configurable_t;

                ///
                /// \brief train the given model starting from the current model parameters
                ///
                virtual trainer_result_t train(
                        const enhancer_t&, const task_t&, const size_t fold, accumulator_t&) const = 0;
        };
}
