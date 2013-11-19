#ifndef MINIBATCH_TRAINER_H
#define MINIBATCH_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // mini-batch trainer: the gradient update is computed
        //      on a random mini-batch of fixed size in epochs.
        //
        // parameters:
        //      opt=lbfgs[,cgd,gd]              - optimization method
        //      iter=4[1,128]                   - maximum number of optimization iterations per epoch
        //      batch=1024[100,10000]           - mini-batch size / epoch
        //      epoch=256[8,1024]               - number of epochs
        //      sample=lwei[,once,rand,lmax]    - sampling strategy for each epoch
        /////////////////////////////////////////////////////////////////////////////////////////

        class minibatch_trainer_t : public trainer_t
        {
        public:

                // constructor
                minibatch_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(minibatch_trainer_t, trainer_t,
                                  "mini-batch trainer, parameters: opt=lbfgs[,cgd,gd],iter=4[1,128],"\
                                  "batch=1024[100,10000],epoch=256[8,1024],"\
                                  "sample=lwei[,once,rand,lmax]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, model_t&) const;

        private:

                // random sampling
                samples_t rand(const samples_t&) const;

                // maximum loss value sampling
                samples_t lmax(const samples_t&, const samples_t&, const task_t&, const loss_t&, const model_t&) const;

                // loss value proportional sampling
                samples_t lwei(const samples_t&, const samples_t&, const task_t&, const loss_t&, const model_t&) const;

                // <loss value, sample index>
                typedef std::pair<scalar_t, size_t>     lvalue_t;
                typedef std::vector<lvalue_t>           lvalues_t;
                lvalues_t make_lvalues(const samples_t&, const task_t&, const loss_t&, const model_t&) const;
                samples_t make_samples(const samples_t&, const lvalues_t&) const;

        private:

                // attributes
                string_t        m_optimizer;
                size_t          m_iterations;
                size_t          m_batchsize;
                size_t          m_epochs;
                scalar_t        m_epsilon;
                string_t        m_sampling;
        };
}

#endif // MINIBATCH_TRAINER_H
