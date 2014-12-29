#pragma once

#include "task.h"
#include "trainer_result.h"

namespace ncv
{
        class loss_t;
        class model_t;
        class trainer_t;
        class sampler_t;
        class accumulator_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<trainer_t>                    trainer_manager_t;
        typedef trainer_manager_t::robject_t            rtrainer_t;
                
        ///
        /// \brief generic trainer: optimizes a model on a given task
        ///
        class trainer_t : public clonable_t<trainer_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                trainer_t(const string_t& configuration)
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
                        const task_t&, const fold_t&, const loss_t&, size_t nthreads, const string_t& criterion, 
                        model_t&) const = 0;
        };
}

