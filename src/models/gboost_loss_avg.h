#pragma once

#include "gboost_loss.h"

namespace nano
{
        ///
        /// \brief given a loss function l(y, t) that measures how well the prediction y matches the target t,
        ///     the empirical expectation of the loss is:
        ///
        ///     L = 1/N sum(l(y_i, t_i), i),
        ///
        ///     over N samples indexed by i.
        ///
        template <typename tweak_learner>
        class gboost_loss_avg_t final : public gboost_loss_t<tweak_learner>
        {
        public:

                using gboost_loss_t<tweak_learner>::m_task;
                using gboost_loss_t<tweak_learner>::m_fold;
                using gboost_loss_t<tweak_learner>::m_loss;
                using gboost_loss_t<tweak_learner>::m_outputs;
                using gboost_loss_t<tweak_learner>::m_wlearner;
                using gboost_loss_t<tweak_learner>::m_residuals;

                gboost_loss_avg_t(const task_t& task, const fold_t& fold, const loss_t& loss) :
                        gboost_loss_t<tweak_learner>(task, fold, loss)
                {
                }

                std::pair<scalar_t, scalar_t> update() override
                {
                        const auto workers = static_cast<tensor_size_t>(tpool_t::instance().workers());

                        tensor1d_t errors(workers);
                        tensor1d_t values(workers);

                        errors.zero();
                        values.zero();

                        loopit(m_task.size(m_fold), [&] (const size_t i, const size_t t)
                        {
                                assert(static_cast<tensor_size_t>(t) < workers);
                                const auto input = m_task.input(m_fold, i);
                                const auto target = m_task.target(m_fold, i);
                                const auto output = m_outputs.tensor(i);

                                errors(t) += m_loss.error(target, output);
                                values(t) += m_loss.value(target, output);
                                m_residuals.tensor(i) = m_loss.vgrad(target, output);
                        });

                        return std::make_pair(reduce(values), reduce(errors));
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
                {
                        assert(x.size() == 1);
                        assert(!gx || gx->size() == 1);

                        const auto workers = static_cast<tensor_size_t>(tpool_t::instance().workers());

                        tensor1d_t values(workers);
                        tensor1d_t vgrads(workers);
                        tensor4d_t outputs(cat_dims(workers, m_task.odims()));

                        values.zero();
                        vgrads.zero();

                        loopit(m_task.size(m_fold), [&] (const size_t i, const size_t t)
                        {
                                assert(static_cast<tensor_size_t>(t) < workers);
                                const auto input = m_task.input(m_fold, i);
                                const auto target = m_task.target(m_fold, i);
                                const auto woutput = m_wlearner.output(input);

                                auto output = outputs.tensor(t);
                                assert(output.dims() == woutput.dims());

                                output.vector() = m_outputs.vector(i) + x(0) * woutput.vector();

                                const auto value = m_loss.value(target, output);
                                values(t) += value;

                                if (gx)
                                {
                                        const auto vgrad = m_loss.vgrad(target, output);
                                        vgrads(t) += vgrad.vector().dot(woutput.vector());
                                }
                        });

                        if (gx)
                        {
                                (*gx)(0) = reduce(vgrads);
                        }
                        return reduce(values);
                }

        private:

                auto size() const
                {
                        return static_cast<scalar_t>(m_task.size(m_fold));
                }

                auto reduce(const tensor1d_t& values) const
                {
                        return values.vector().sum() / size();
                }
        };
}
