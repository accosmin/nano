#ifndef NANOCV_TRAINER_H
#define NANOCV_TRAINER_H

#include "task.h"
#include "loss.h"
#include "model.h"
#include "trainer_state.h"

namespace ncv
{
        // manage trainers (register new ones, query and clone them)
        class trainer_t;
        typedef manager_t<trainer_t>            trainer_manager_t;
        typedef trainer_manager_t::robject_t    rtrainer_t;

        // (batch) train a model on the given training & validation samples
        bool train(
                const task_t&, const samples_t& tsamples, const samples_t& vsamples, const loss_t&,
                const string_t& optimizer, scalar_t epsilon, size_t iterations, size_t nthreads,
                model_t& model, trainer_state_t& state, bool verbose = true);

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
