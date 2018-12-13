#pragma once

#include "gboost.h"
#include "cortex.h"

namespace nano
{
        // todo: generalize it to use other features (e.g products of inputs, LBPs|HOGs)!
        //
        class task_t;
        class ibstream_t;
        class obstream_t;

        ///
        /// \brief a stump is a weak learner that compares the value of a selected feature with a threshold:
        ///     stump(x) = outputs(0) if x(feature) < threshold else v1(0)
        ///
        class NANO_PUBLIC wlearner_stump_t
        {
        public:

                ///
                /// \brief default constructor
                ///
                wlearner_stump_t() = default;

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
                /// \brief fit its parameters to the given gradients
                ///
                void fit(const task_t&, const fold_t&, const tensor4d_t& gradients, const wlearner_type);

                ///
                /// \brief scale the outputs by the given factor
                ///
                void scale(const scalar_t factor)
                {
                        assert(factor >= 0);
                        m_outputs.array() *= factor;
                }

                ///
                /// \brief scale the outputs by the given factors
                ///
                void scale(const vector_t& factors)
                {
                        assert(2 * factors.size() == m_outputs.size());
                        m_outputs.array(0) *= factors.array();
                        m_outputs.array(1) *= factors.array();
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

                scalar_t fit(const task_t&, const fold_t&, const tensor4d_t& gradients,
                        const tensor_size_t feature, const wlearner_type);

                template <typename ttensor, typename tarray>
                static auto fit_value(const int cnt, const ttensor& res1, const ttensor& res2, const tarray& outputs)
                {
                        return (cnt * outputs.square() - 2 * outputs * res1.array() + res2.array()).sum();
                }

                template <typename ttensor, typename tarray>
                void try_fit(
                        const int cnt_neg, const ttensor& res_neg1, const ttensor& res_neg2, const tarray& outputs_neg,
                        const int cnt_pos, const ttensor& res_pos1, const ttensor& res_pos2, const tarray& outputs_pos,
                        const tensor_size_t feature, const scalar_t threshold, scalar_t& value)
                {
                        const auto tvalue =
                                fit_value(cnt_neg, res_neg1, res_neg2, outputs_neg) +
                                fit_value(cnt_pos, res_pos1, res_pos2, outputs_pos);

                        if (tvalue < value)
                        {
                                value = tvalue;
                                m_feature = feature;
                                m_threshold = threshold;
                                m_outputs.array(0) = outputs_neg;
                                m_outputs.array(1) = outputs_pos;
                        }
                }

                static scalars_t fvalues(const task_t&, const fold_t&, const tensor_size_t feature);
                static scalars_t thresholds(const scalars_t& fvalues);

        private:

                // attributes
                tensor_size_t   m_feature{0};   ///< index of the selected feature
                scalar_t        m_threshold{0}; ///< threshold
                tensor4d_t      m_outputs;      ///< (2, #outputs) - predictions below and above the threshold
        };
}
