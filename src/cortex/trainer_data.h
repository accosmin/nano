#pragma once

#include "cortex/sampler.h"
#include "cortex/optimizer.h"

namespace cortex
{
        class task_t;
        class loss_t;
        class accumulator_t;

        ///
        /// \brief stores all required buffers to train a model
        ///
        struct trainer_data_t
        {
                ///
                /// \brief constructor
                ///
                trainer_data_t(const task_t& task,
                               const sampler_t& tsampler,
                               const sampler_t& vsampler,
                               const loss_t& loss,
                               const vector_t& x0,
                               accumulator_t& lacc,
                               accumulator_t& gacc);

                ///
                /// \brief change the regularization weight
                ///
                void set_lambda(scalar_t lambda);

                ///
                /// \brief get the regularization weight
                ///
                scalar_t lambda() const;

                ///
                /// \brief compute the epoch size (# iterations per epoch) given the batch size in samples
                ///
                size_t epoch_size(const size_t batchsize) const;

                // attributes
                const task_t&           m_task;                 ///< 
                sampler_t               m_tsampler;             ///< training samples
                sampler_t               m_vsampler;             ///< validation samples

                const loss_t&           m_loss;                 ///< base loss function
                const vector_t&         m_x0;                   ///< initial parameters

                accumulator_t&          m_lacc;                 ///< cumulated loss value
                accumulator_t&          m_gacc;                 ///< cumulated loss gradient
        };

        ///
        /// \brief regularization factors to tune
        ///
        scalars_t tunable_lambdas();

        ///
        /// \brief dimension operator
        ///
        opt_opsize_t make_opsize(const trainer_data_t& data);

        ///
        /// \brief cumulated loss value operator
        ///
        opt_opfval_t make_opfval(const trainer_data_t& data);

        ///
        /// \brief cumulated loss gradient operator
        ///
        opt_opgrad_t make_opgrad(const trainer_data_t& data);
}

