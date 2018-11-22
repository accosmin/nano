#pragma once

#include "gboost_loss.h"

namespace nano
{
        ///
        /// \brief empirical expection of the loss.
        ///
        template <typename tweak_learner>
        class gboost_loss_avg_t final : public gboost_loss_t<tweak_learner>
        {
        public:

                gboost_loss_avg_t(const task_t& task, const fold_t& fold, const loss_t& loss) :
                        gboost_loss_t(task, fold, loss)
                {
                }

                std::pair<scalar_t, scalar_t> update() override
                {
                        const auto& tpool = tpool_t::instance();

                        tensor1d_t errors(tpool.workers());
                        tensor1d_t values(tpool.workers());

                        errors.zero();
                        values.zero();

                        loopit(m_task.size(m_fold), [&] (const size_t i, const size_t t)
                        {
                                assert(t < tpool.workers());
                                const auto input = m_task.input(m_fold, i);
                                const auto target = m_task.target(m_fold, i);
                                const auto output = m_outputs.tensor(i);

                                errors(t) += m_loss.error(target, output);
                                values(t) += m_loss.value(target, output);
                                m_residuals.tensor(i) = loss.vgrad(target, output);
                        });

                        const auto div = static_cast<scalar_t>(m_task.size(m_fold));
                        return std::make_pair(values.vector().sum() / div, errors.vector().sum() / div);
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
        };
}
