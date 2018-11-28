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

                gboost_loss_avg_t(const task_t& task, const fold_t& fold, const loss_t& loss) :
                        gboost_loss_t<tweak_learner>(task, fold, loss)
                {
                }

                std::pair<scalar_t, scalar_t> update() override
                {
                        auto errors = this->tpool1d();
                        auto values = this->tpool1d();

                        loopit(this->m_task.size(this->m_fold), [&] (const size_t i, const size_t t)
                        {
                                const auto input = this->m_task.input(this->m_fold, i);
                                const auto target = this->m_task.target(this->m_fold, i);
                                const auto output = this->m_outputs.tensor(i);

                                errors(t) += this->m_loss.error(target, output);
                                values(t) += this->m_loss.value(target, output);
                                this->m_residuals.tensor(i) = this->m_loss.vgrad(target, output);
                        });

                        return std::make_pair(reduce(values), reduce(errors));
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
                {
                        assert(x.size() == 1);
                        assert(!gx || gx->size() == 1);

                        auto values = this->tpool1d();
                        auto vgrads = this->tpool1d();

                        loopit(this->m_task.size(this->m_fold), [&] (const size_t i, const size_t t)
                        {
                                const auto input = this->m_task.input(this->m_fold, i);
                                const auto target = this->m_task.target(this->m_fold, i);
                                const auto woutput = this->m_wlearner.output(input);

                                tensor3d_t output(this->m_task.odims());
                                assert(output.dims() == woutput.dims());

                                output.vector() = this->m_outputs.vector(i) + x(0) * woutput.vector();

                                const auto value = this->m_loss.value(target, output);
                                values(t) += value;

                                if (gx)
                                {
                                        const auto vgrad = this->m_loss.vgrad(target, output);
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

                auto reduce(const tensor1d_t& values) const
                {
                        return values.vector().sum() / this->size();
                }
        };
}
