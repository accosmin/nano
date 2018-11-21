#pragma once

#include "loss.h"
#include "task.h"
#include "function.h"
#include "core/stats.h"
#include "core/tpool.h"

namespace nano
{
        ///
        /// \brief squared empirical expection of the loss with a scaled variance of the loss.
        ///     see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
        ///
        class gboost_vloss_t
        {
        public:

                gboost_vloss_t(
                        const task_t& task, const fold_t& fold, const loss_t& loss, const tensor4d_t& outputs,
                        const scalar_t lambda) :
                        m_task(task), m_fold(fold), m_loss(loss), m_outputs(outputs), m_lambda(lambda),
                        m_residuals(outputs.dims())
                {
                        assert(m_outputs.dims() == cat_dims(task.size(fold), task.odims());
                }

                void update()
                {
                        // todo: L = lambda * N * S2 + (1 - lambda) * S1^2
                        // todo: Gi = 2 * lambda * N * li * gi + 2 * (1 - lambda) * S1 * gi

                        const auto& tpool = tpool_t::instance();

                        std::vector<stats_t> errors(tpool.workers());
                        std::vector<stats_t> values(tpool.workers());
                        loopit(m_task.size(m_fold), [&] (const size_t i, const size_t t)
                        {
                                assert(t < tpool.workers());
                                const auto input = m_task.input(m_fold, i);
                                const auto target = m_task.target(m_fold, i);
                                const auto output = m_outputs.tensor(i);

                                errors[t](m_loss.error(target, output));
                                values[t](m_loss.value(target, output));
                                m_residuals.tensor(i) = loss.vgrad(target, output);
                        });

                        m_estats.clear();
                        m_vstats.clear();
                        for (size_t t = 0; t < tpool.workers(); ++ t)
                        {
                                m_estats(errors[t]);
                                m_vstats(values[t]);
                        }
                }

                const auto& residuals() const { return m_residuals; }
                const auto& vstats() const { return m_vstats; }
                const auto& estats() const { return m_estats; }

        private:

                // attributes
                const task_t&           m_task;         ///< given task
                fold_t                  m_fold;         ///< given fold
                const loss_t&           m_loss;         ///< given loss
                const tensor4d_t&       m_outputs;      ///< given outputs for each sample
                tensor4d_t              m_residuals;    ///<
                stats_t                 m_vstats;       ///< loss value statistics
                stats_t                 m_estats;       ///< error value statistics
        };
}
