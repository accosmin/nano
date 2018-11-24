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
                        function_t("gboost-loss", 1, 1, 1, convexity::no),
                        m_task(task), m_fold(fold), m_loss(loss),
                        m_outputs(cat_dims(task.size(fold), task.odims())),
                        m_residuals(cat_dims(task.size(fold), task.odims()))
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
                                m_outputs.array(i) += wlearner.output(input);
                        });
                }

                ///
                /// \brief update its internal state (e.g. residuals) and
                ///     return the loss value and the average error rate
                ///
                virtual std::pair<scalar_t, scalar_t> update() = 0;

                ///
                /// \brief compute the line-search function value and gradient
                ///
                virtual scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const = 0;

                ///
                /// \brief access functions
                ///
                const auto& wlearner() const { return m_wlearner; }
                const auto& residuals() const { return m_residuals; }

        protected:

                // attributes
                const task_t&   m_task;         ///< given task
                fold_t          m_fold;         ///< given fold
                const loss_t&   m_loss;         ///< given loss
                tensor4d_t      m_outputs;      ///< output/prediction for each sample
                tensor4d_t      m_residuals;    ///< gradient/residual for each sample
                tweak_learner   m_wlearner;     ///< current weak learner
        };
}
