#pragma once

#include "gboost.h"
#include "wlearner.h"

namespace nano
{
        // todo: generalize it to use other features (e.g products of inputs, LBPs|HOGs)!
        //
        class ibstream_t;
        class obstream_t;

        ///
        /// \brief
        ///
        enum class stump_type
        {
                real,           ///< output \in R (no restriction)
                discrete,       ///< output \in {-1, +1} (useful for classification to reduce overfitting)
        };

        ///
        /// \brief a stump is a weak learner that compares the value of a selected feature with a threshold:
        ///     stump(x) = outputs(0) if x(feature) < threshold else outputs(1)
        ///
        template <stump_type type>
        class wlearner_stump_t : public wlearner_t
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
                /// \brief fit its parameters to the given gradients and feature
                ///
                scalar_t fit(const task_t&, const fold_t&, const tensor4d_t& gradients, const indices_t&,
                        const tensor_size_t feature);

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
                        assert(factors.minCoeff() >= 0);
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
                auto threshold() const { return m_threshold; }
                const auto& outputs() const { return m_outputs; }

        private:

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

        private:

                // attributes
                scalar_t        m_threshold{0}; ///< threshold
                tensor4d_t      m_outputs;      ///< (2, #outputs) - predictions below and above the threshold
        };

        template <stump_type type>
        std::ostream& operator<<(std::ostream& os, const wlearner_stump_t<type>& stump)
        {
                return os << "stump=(f=" << stump.feature() << ",t=" << stump.threshold() << ")";
        }

        using wlearner_real_stump_t = wlearner_stump_t<stump_type::real>;
        using wlearner_discrete_stump_t = wlearner_stump_t<stump_type::discrete>;
}
