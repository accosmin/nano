#pragma once

#include "loss.h"
#include "task.h"
#include "core/stats.h"
#include "core/tpool.h"

namespace nano
{
        ///
        /// \brief loss function used by GradientBoosting algorithms.
        ///
        class gboost_loss_t
        {
        public:

                ///
                /// \brief constructor
                ///
                gboost_loss_t(const task_t& task, const fold_t& fold, const loss_t& loss) :
                        m_task(task), m_fold(fold), m_loss(loss),
                        m_outputs(cat_dims(task.size(fold), task.odims())),
                        m_residuals(cat_dims(task.size(fold), task.odims()))
                {
                        m_outputs.zero();
                }

                ///
                /// \brief destructor
                ///
                virtual ~gboost_loss_t() = default;

                ///
                /// \brief compute the loss value and the residuals for each sample
                ///
                virtual scalar_t compute() = 0;

                ///
                /// \brief update the outputs with the predictions of a weak learner
                ///
                template <typename tweak_learner>
                void add_weak_learner(const tweak_learner& wlearner)
                {
                        loopi(m_task.size(fold), [&] (const size_t i)
                        {
                                const auto input = m_task.input(m_fold, i);
                                m_outputs.array(i) += wlearner.output(input);
                        });
                }

                ///
                /// \brief access functions
                ///
                const auto& residuals() const { return m_residuals; }
                const auto& vstats() const { return m_vstats; }
                const auto& estats() const { return m_estats; }

        protected:

                // attributes
                const task_t&   m_task;         ///< given task
                fold_t          m_fold;         ///< given fold
                const loss_t&   m_loss;         ///< given loss
                tensor4d_t      m_outputs;      ///< output/prediction for each sample
                tensor4d_t      m_residuals;    ///< gradient/residual for each sample
                stats_t         m_vstats;       ///< loss value statistics for all samples
                stats_t                 m_estats;       ///< error value statistics for all samples
        };
}
