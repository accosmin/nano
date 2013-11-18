#ifndef NANOCV_TRAINER_H
#define NANOCV_TRAINER_H

#include "task.h"
#include "loss.h"
#include "model.h"

namespace ncv
{
        // manage trainers (register new ones, query and clone them)
        class trainer_t;
        typedef manager_t<trainer_t>            trainer_manager_t;
        typedef trainer_manager_t::robject_t    rtrainer_t;

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
                virtual bool train(const task_t&, const fold_t&, const loss_t&, model_t&) const = 0;

                // prune samples
                static samples_t prune_annotated(const task_t&, const samples_t&);

                // compute loss value & gradient (given the model)
                //      (single & multi-threaded versions)
                static scalar_t value(const task_t&, const sample_t&, const loss_t&, const model_t&);
                static scalar_t value_st(const task_t&, const samples_t&, const loss_t&, const model_t&);
                static scalar_t value_mt(const task_t&, const samples_t&, const loss_t&, const model_t&);
                static scalar_t vgrad_st(const task_t&, const samples_t&, const loss_t&, const model_t&, vector_t&);
                static scalar_t vgrad_mt(const task_t&, const samples_t&, const loss_t&, const model_t&, vector_t&);
        };
}

#endif // NANOCV_TRAINER_H
