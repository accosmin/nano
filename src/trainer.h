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

        // prune samples
        samples_t prune_annotated(const task_t&, const samples_t&);

        // split samples into:
        //      training        - (100 - vpercentage)%
        //      validation      - vpercentage%
        void split_train_valid(const samples_t&, size_t vpercentage, samples_t& tsamples, samples_t& vsamples);

        // compute loss value & gradient (given a model and some samples)
        //      (single & multi-threaded versions)
        scalar_t lvalue(const task_t&, const sample_t&, const loss_t&, const model_t&);
        scalar_t lvgrad(const task_t&, const sample_t&, const loss_t&, const model_t&);
        scalar_t lvalue_st(const task_t&, const samples_t&, const loss_t&, const model_t&);
        scalar_t lvalue_mt(const task_t&, const samples_t&, const loss_t&, size_t nthreads, const model_t&);
        scalar_t lvgrad_st(const task_t&, const samples_t&, const loss_t&, const model_t&, vector_t&);
        scalar_t lvgrad_mt(const task_t&, const samples_t&, const loss_t&, size_t nthreads, const model_t&, vector_t&);

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

//        protected:

//                //
//                bool optimize(const task_t&, const samples_t& tsamples, const samples_t& vsamples, const loss_t&,
//                              const string_t& optimizer, scalar_t epsilon, size_t iterations, size_t nthreads,
//                              model_t&) const;
        };
}

#endif // NANOCV_TRAINER_H
