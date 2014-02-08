#ifndef NANOCV_TRAINER_H
#define NANOCV_TRAINER_H

#include "task.h"
#include "loss.h"
#include "model.h"

namespace ncv
{
        class trainer_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<trainer_t>                    trainer_manager_t;
        typedef trainer_manager_t::robject_t            rtrainer_t;

        /////////////////////////////////////////////////////////////////////////////////////////
        // generic trainer:
        //      optimizes a model on a given task.
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class trainer_t : public clonable_t<trainer_t>
        {
        public:

                // destructor
                virtual ~trainer_t() {}

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const = 0;
        };
}

#endif // NANOCV_TRAINER_H
