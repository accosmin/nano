#pragma once

#include "gboost_loss.h"

namespace nano
{
        ///
        /// \brief given a loss function l(y, t) that measures how well the prediction y matches the target t,
        ///     the squared empirical expection of the loss regularized with its variance is:
        ///
        ///     L = sum(l(y_i, t_i), i)^2 + lambda * sum((l(y_i, t_i) - l(y_j, t_j))^2, i<j),
        ///       = lambda * sum(l(y_i, t_i)^2, i) + (1 - lambda) * sum(l(y_i, t_i)^2, i) / N
        ///
        ///     over N samples indexed by i.
        ///
        ///     see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
        ///
        template <typename tweak_learner>
        class gboost_loss_var_t final : public gboost_loss_t<tweak_learner>
        {
        public:

                gboost_loss_var_t(const task_t& task, const fold_t& fold, const loss_t& loss, const scalar_t lambda) :
                        gboost_loss_t<tweak_learner>(task, fold, loss),
                        m_lambda(lambda),
                        m_vresiduals(cat_dims(task.size(this->m_fold), task.odims()))
                {
                }

                std::pair<scalar_t, scalar_t> update() override
                {
                        auto errors = this->tpool1d();
                        auto values1 = this->tpool1d();
                        auto values2 = this->tpool1d();

                        loopit(this->m_task.size(this->m_fold), [&] (const size_t i, const size_t t)
                        {
                                const auto input = this->m_task.input(this->m_fold, i);
                                const auto target = this->m_task.target(this->m_fold, i);
                                const auto output = this->m_outputs.tensor(i);

                                const auto error = this->m_loss.error(target, output);
                                const auto value = this->m_loss.value(target, output);
                                const auto vgrad = this->m_loss.vgrad(target, output);

                                errors(t) += error;
                                values1(t) += value;
                                values2(t) += value * value;
                                this->m_residuals.tensor(i) = vgrad;
                                m_vresiduals.vector(i) = value * vgrad.vector();
                        });

                        const auto div = static_cast<scalar_t>(this->m_task.size(this->m_fold));
                        const auto sum1 = values1.vector().sum();

                        loopi(this->m_task.size(this->m_fold), [&] (const size_t i)
                        {
                                this->m_residuals.vector(i) =
                                        2 * m_lambda * m_vresiduals.vector(i) +
                                        2 * (1 - m_lambda) * sum1 / div * this->m_residuals.vector(i);
                        });

                        return std::make_pair(reduce_value(values1, values2), reduce_error(errors));
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
                {
                        assert(x.size() == 1);
                        assert(!gx || gx->size() == 1);

                        auto values1 = this->tpool1d();
                        auto values2 = this->tpool1d();
                        auto vgrads1 = this->tpool1d();
                        auto vgrads2 = this->tpool1d();

                        loopit(this->m_task.size(this->m_fold), [&] (const size_t i, const size_t t)
                        {
                                const auto input = this->m_task.input(this->m_fold, i);
                                const auto target = this->m_task.target(this->m_fold, i);
                                const auto woutput = this->m_wlearner.output(input);

                                tensor3d_t output(this->m_task.odims());
                                assert(output.dims() == woutput.dims());

                                output.vector() = this->m_outputs.vector(i) + x(0) * woutput.vector();

                                const auto value = this->m_loss.value(target, output);
                                values1(t) += value;
                                values2(t) += value * value;

                                if (gx)
                                {
                                        const auto vgrad = this->m_loss.vgrad(target, output);
                                        vgrads1(t) += vgrad.vector().dot(woutput.vector());
                                        vgrads2(t) += value * vgrad.vector().dot(woutput.vector());
                                }
                        });

                        if (gx)
                        {
                                (*gx)(0) = reduce_vgrad(values1, vgrads1, vgrads2);
                        }
                        return reduce_value(values1, values2);
                }

        private:

                auto reduce_error(const tensor1d_t& errors) const
                {
                        return errors.vector().sum() / this->size();
                }

                auto reduce_value(const tensor1d_t& values1, const tensor1d_t& values2) const
                {
                        return  (1 - m_lambda) * nano::square(values1.vector().sum()) +
                                m_lambda * this->size() * values2.vector().sum();
                }

                auto reduce_vgrad(const tensor1d_t& values1, const tensor1d_t& vgrads1, const tensor1d_t& vgrads2) const
                {
                        return  2 * (1 - m_lambda) * values1.vector().sum() * vgrads1.vector().sum() +
                                2 * m_lambda * this->size() * vgrads2.vector().sum();
                }

        private:

                // attributes
                scalar_t        m_lambda;       ///< regularization term for the variance term
                tensor4d_t      m_vresiduals;   ///< residuals/gradients * loss value for each sample
        };
}
