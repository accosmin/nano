#pragma once

#include "types.h"

namespace ncv
{
        class task_t;
        class loss_t;
        class sampler_t;
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
                
                // attributes
                const task_t&           m_task;                 ///< 
                const sampler_t&        m_tsampler;             ///< training samples
                const sampler_t&        m_vsampler;             ///< validation samples

                const loss_t&           m_loss;                 ///< base loss function
                const vector_t&         m_x0;                   ///< initial parameters

                accumulator_t&          m_lacc;                 ///< cumulated loss value
                accumulator_t&          m_gacc;                 ///< cumulated loss gradient
        };
}

