#pragma once

#include "task.h"
#include "core/tpool.h"

namespace nano
{
        // todo: generalize it to use other features (e.g products of inputs, LBPs|HOGs)!

        ///
        /// \brief a weak learner is performing a transformation of the selected feature value.
        ///
        class wlearner_t
        {
        public:

                ///
                /// \brief default constructor
                ///
                wlearner_t() = default;

                ///
                /// \brief fit its parameters to the given gradients
                ///
                template <typename twlearner>
                static void fit(const task_t& task, const fold_t& fold,
                        const tensor4d_t& gradients, const indices_t& indices, twlearner& learner)
                {
                        assert(cat_dims(task.size(fold), task.odims()) == gradients.dims());

                        const auto& tpool = tpool_t::instance();

                        std::vector<twlearner> learners(tpool.workers());
                        scalars_t tvalues(tpool.workers(), std::numeric_limits<scalar_t>::max());

                        loopit(nano::size(task.idims()), [&] (const tensor_size_t feature, const size_t t)
                        {
                                twlearner learner;
                                const auto value = learner.fit(task, fold, gradients, indices, feature);
                                if (std::isfinite(value) && value < tvalues[t])
                                {
                                        tvalues[t] = value;
                                        std::swap(learners[t], learner);
                                }
                        });

                        learner = learners[std::min_element(tvalues.begin(), tvalues.end()) - tvalues.begin()];
                }

                ///
                /// \brief change the selected feature
                ///
                auto& feature(const tensor_size_t feature)
                {
                        assert(feature >= 0);
                        m_feature = feature;
                        return *this;
                }

                ///
                /// \brief returns the selected feature
                ///
                auto feature() const { return m_feature; }

        protected:

                // attributes
                tensor_size_t   m_feature{0};   ///< index of the selected feature
        };
}
