#pragma once

#include "loss.h"
#include "task.h"
#include "function.h"
#include "core/tpool.h"

namespace nano
{
        ///
        /// \brief loss function used by GradientBoosting for:
        ///     - feature selection using residuals (~ loss gradients per sample)
        ///     - and 1D line-search for scaling the selected feature (aka weak learner)
        ///
        template <typename tweak_learner>
        class gboost_loss_t : public function_t
        {
        public:

                ///
                /// \brief constructor
                ///
                gboost_loss_t(const task_t& task, const fold_t& fold, const loss_t& loss) :
                        function_t("gboost-loss", nano::size(task.odims()), 1, 1, convexity::no),
                        m_task(task), m_fold(fold), m_loss(loss),
                        m_outputs(cat_dims(task.size(fold), task.odims()))
                {
                        m_outputs.zero();
                }

                ///
                /// \brief use the given weak learner for line-search
                ///
                void wlearner(const tweak_learner& wlearner)
                {
                        m_wlearner = wlearner;
                }

                ///
                /// \brief update the outputs with the predictions of the given weak learner
                ///
                void add_wlearner(const tweak_learner& wlearner)
                {
                        loopi(m_task.size(m_fold), [&] (const size_t i)
                        {
                                const auto input = m_task.input(m_fold, i);
                                m_outputs.vector(i) += wlearner.output(input).vector();
                        });
                }

                ///
                /// \brief compute the line-search function value and gradient
                ///
                /// virtual scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const = 0;

                ///
                /// \brief compute and return the average error
                ///
                scalar_t error() const;

                ///
                /// \brief compute and return the loss value
                ///
                virtual scalar_t value() const = 0;

                ///
                /// \brief compute and return the residuals (gradients per sample)
                ///
                virtual const tensor4d_t& residuals() = 0;

                ///
                /// \brief returns the current outputs/predictions
                ///
                const auto& outputs() const { return m_outputs; }

                ///
                /// \brief returns the current weak learner
                ///
                const auto& wlearner() const { return m_wlearner; }

        protected:

                auto fold_size() const
                {
                        return static_cast<scalar_t>(m_task.size(m_fold));
                }

                auto workers() const
                {
                        return static_cast<tensor_size_t>(tpool_t::instance().workers());
                }

                auto tpool1d() const
                {
                        tensor1d_t buffer(workers());
                        buffer.zero();
                        return buffer;
                }

                auto tpool2d(const tensor_size_t size) const
                {
                        tensor2d_t buffer(workers(), size);
                        buffer.zero();
                        return buffer;
                }

                auto reduce(const tensor1d_t& values) const
                {
                        return values.vector().sum() / fold_size();
                }

                auto reduce(const tensor2d_t& values) const
                {
                        return values.matrix().colwise().sum() / fold_size();
                }

        protected:

                // attributes
                const task_t&   m_task;         ///< given task
                fold_t          m_fold;         ///< given fold
                const loss_t&   m_loss;         ///< given loss
                tensor4d_t      m_outputs;      ///< output/prediction for each sample
                tweak_learner   m_wlearner;     ///< current weak learner
        };

        template <typename tweak_learner>
        scalar_t gboost_loss_t<tweak_learner>::error() const
        {
                auto errors = tpool1d();

                loopit(m_task.size(m_fold), [&] (const size_t i, const size_t t)
                {
                        const auto input = m_task.input(m_fold, i);
                        const auto target = m_task.target(m_fold, i);
                        const auto output = m_outputs.tensor(i);

                        errors(t) += m_loss.error(target, output);
                });

                return reduce(errors);
        }
}
