#pragma once

#include "gboost_loss.h"

namespace nano
{
        ///
        /// \brief squared empirical expection of the loss with a scaled variance of the loss.
        ///     see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
        ///
        template <typename tweak_learner>
        class gboost_loss_var_t final : public gboost_loss_t<tweak_learner>
        {
        public:

                gboost_loss_var_t(const task_t& task, const fold_t& fold, const loss_t& loss, const scalar_t lambda) :
                        gboost_loss_t(task, fold, loss),
                        m_lambda(lambda),
                        m_vresiduals(cat_dims(task.size(m_fold), task.odims()))
                {
                }

                std::pair<scalar_t, scalar_t> update() override
                {
                        // todo: L = lambda * S2 + (1 - lambda) * S1^2 / N
                        // todo: Gi = 2 * lambda * li * gi + 2 * (1 - lambda) * S1/N * gi

                        const auto& tpool = tpool_t::instance();

                        tensor1d_t errors(tpool.workers());
                        tensor1d_t values1(tpool.workers());
                        tensor1d_t values2(tpool.workers());

                        errors.zero();
                        values1.zero();
                        values2.zero();

                        loopit(m_task.size(m_fold), [&] (const size_t i, const size_t t)
                        {
                                assert(t < tpool.workers());
                                const auto input = m_task.input(m_fold, i);
                                const auto target = m_task.target(m_fold, i);
                                const auto output = m_outputs.tensor(i);

                                const auto error = m_loss.error(target, output);
                                const auto value = m_loss.value(target, output);
                                const auto vgrad = m_loss.vgrad(target, output);

                                errors(t) += error;
                                values1(t) += value;
                                values2(t) += value * value;
                                m_residuals.tensor(i) = vgrad;
                                m_vresiduals.vector(i) = value * vgrad.vector();
                        });

                        const auto div = static_cast<scalar_t>(m_task.size(m_fold));
                        const auto sum1 = values1.vector().sum();
                        const auto sum2 = values2.vector().sum();

                        loopi(m_task.size(m_fold), [&] (const size_t i)
                        {
                                m_residuals.vector(i) =
                                        2 * m_lambda * m_vresiduals.vector(i) +
                                        2 * (1 - m_lambda) * sum1 / div * m_residuals.vector(i);
                        });

                        const auto value = m_lambda * sum2 + (1 - lambda) * nano::square(sum1) / div;
                        const auto error = errors.vector().sum() / div;

                        return std::make_pair(value, error);
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
                {
                        assert(x.size() == 1);
                        assert(!gx || gx->size() == 1);

                        const auto& tpool = tpool_t::instance();

                        tensor1d_t values(tpool.workers());
                        tensor1d_t vgrads(tpool.workers());
                        tensor4d_t outputs(cat_dims(tpool.workers(), m_task.m_odims()));

                        values.zero();
                        vgrads.zero();

                        // todo: finish this!
                        // todo: L = lambda * S2 + (1 - lambda) * S1^2 / N
                        // todo: Gi = 2 * lambda * li * gi + 2 * (1 - lambda) * S1/N * gi

                        loopit(m_task.size(m_fold), [&] (const size_t i, const size_t t)
                        {
                                assert(t < tpool.workers());
                                const auto input = m_task.input(m_fold, i);
                                const auto target = m_task.target(m_fold, i);
                                const auto output = outputs.tensor(t);
                                const auto woutput = m_wlearner.output(input);
                                assert(output.dims() == woutput.dims());

                                output.vector() = m_outputs.vector(i) + x(0) * woutput;
                                values(t) += m_loss.value(target, output);

                                if (gx)
                                {
                                        const auto vgrad = m_loss.vgrad(target, output);
                                        vgrads(t) += vgrad.vector().dot(woutput.vector());
                                }
                        });

                        const auto div = static_cast<scalar_t>(m_task.size(m_fold));
                        if (gx)
                        {
                                *gx(0) = vgrads.vector().sum() / div;
                        }
                        return values.vector().sum() / div;
                }

        private:

                // attributes
                scalar_t        m_lambda;       ///< regularization term for the variance term
                tensor4d_t      m_vresiduals;   ///< residuals/gradients * loss value for each sample
        };
}
