#pragma once

#include "task.h"
#include "trainer_result.h"

namespace cortex
{
        class loss_t;
        class model_t;
        class trainer_t;
        class sampler_t;
        class criterion_t;
        class accumulator_t;

        ///
        /// \brief stores registered prototypes
        ///
        using trainer_manager_t = manager_t<trainer_t>;
        using rtrainer_t = trainer_manager_t::trobject;

        NANOCV_PUBLIC trainer_manager_t& get_trainers();

        ///
        /// \brief generic trainer: optimizes a model on a given task
        ///
        class NANOCV_PUBLIC trainer_t : public clonable_t<trainer_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit trainer_t(const string_t& configuration)
                        :       clonable_t<trainer_t>(configuration)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~trainer_t() {}

                ///
                /// \brief train the given model
                ///
                virtual trainer_result_t train(
                        const task_t&, const fold_t&, const loss_t&, size_t nthreads, const criterion_t& criterion,
                        model_t&) const = 0;
        };
}

