#pragma once

#include "factory.h"
#include "configurable.h"
#include "trainer_result.h"

namespace nano
{
        class loss_t;
        class task_t;
        class model_t;
        class accumulator_t;

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

                ///
                /// \brief tune its hyper-parameters to address the given task
                ///
                virtual void tune(const task_t&, const size_t fold, accumulator_t&) = 0;

                ///
                /// \brief train the given model starting from the current model parameters
                ///
                virtual trainer_result_t train(const task_t&, const size_t fold, accumulator_t&) const = 0;
        };
}
