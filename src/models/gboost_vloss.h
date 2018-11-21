#pragma once

#include "gboost_loss.h"

namespace nano
{
        ///
        /// \brief squared empirical expection of the loss with a scaled variance of the loss.
        ///     see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
        ///
        class gboost_vloss_t final : public gboost_loss_t
        {
        public:

                gboost_vloss_t(const task_t& task, const fold_t& fold, const loss_t& loss, const scalar_t lambda) :
                        gboost_loss_t(task, fold, loss),
                        m_lambda(lambda)
                {
                }

                void compute() override
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

        private:

                // attributes
                scalar_t        m_lambda;       ///< regularization term for the variance term
        };
}
