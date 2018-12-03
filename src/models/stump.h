#pragma once

#include "gboost.h"
#include "cortex.h"

namespace nano
{
        // todo: generalize stump_t to use other features (e.g products of inputs, LBPs|HOGs)
        //
        class task_t;
        class ibstream_t;
        class obstream_t;

        ///
        /// \brief a stump is a weak learner that compares the value of a selected feature with a threshold:
        ///     stump(x) = outputs(0) if x(feature) < threshold else v1(0)
        ///
        class stump_t
        {
        public:

                ///
                /// \brief default constructor
                ///
                stump_t() = default;

                ///
                /// \brief compute the output/prediction given a 3D tensor input
                ///
                template <typename ttensor3d>
                auto output(const ttensor3d& input) const
                {
                        assert(m_outputs.size<0>() == 2);
                        assert(m_feature >= 0 && m_feature < input.size());
                        const auto oindex = input(m_feature) < m_threshold ? 0 : 1;
                        return m_outputs.tensor(oindex);
                }

                ///
                /// \brief fit the stump to the given residuals
                ///
                void fit(const task_t&, const fold_t&, const tensor4d_t& residuals, const wlearner_type);

                ///
                /// \brief scale the outputs by the given factor
                ///
                void scale(const scalar_t factor)
                {
                        m_outputs.array() *= factor;
                }

                ///
                /// \brief serialize to disk
                ///
                bool load(ibstream_t&);
                bool save(obstream_t&) const;

                ///
                /// \brief change parameters
                ///
                auto& feature(const tensor_size_t feature)
                {
                        assert(feature >= 0);
                        m_feature = feature;
                        return *this;
                }
                auto& threshold(const scalar_t threshold)
                {
                        m_threshold = threshold;
                        return *this;
                }
                auto& outputs(const tensor4d_t& outputs)
                {
                        assert(outputs.size<0>() == 2);
                        m_outputs = outputs;
                        return *this;
                }

                ///
                /// \brief access functions
                ///
                auto feature() const { return m_feature; }
                auto threshold() const { return m_threshold; }
                const auto& outputs() const { return m_outputs; }

        private:

                scalar_t fit(const task_t& task, const tensor4d_t& residuals,
                        const tensor_size_t feature, const scalars_t& fvalues, const scalars_t& thresholds,
                        const wlearner_type);

                static scalars_t fvalues(const task_t&, const fold_t&, const tensor_size_t feature);
                static scalars_t thresholds(const scalars_t& fvalues);

        private:

                // attributes
                tensor_size_t   m_feature{0};   ///< index of the selected feature
                scalar_t        m_threshold{0}; ///< threshold
                tensor4d_t      m_outputs;      ///< (2, #outputs) - predictions below and above the threshold
        };

        using stumps_t = std::vector<stump_t>;
}
