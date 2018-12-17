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
        /// \brief
        ///
        enum class table_type
        {
                real,           ///< output \in R (no restriction)
                discrete,       ///< output \in {-1, +1} (useful for classification to reduce overfitting)
        };

        ///
        /// \brief a (look-up) table is a weak learner that returns the output indexed by the selected feature:
        ///     table(x) = outputs(static_cast<int>(x(feature)))
        ///
        template <table_type type>
        class wlearner_table_t
        {
        public:

                ///
                /// \brief default constructor
                ///
                wlearner_table_t() = default;

                ///
                /// \brief compute the output/prediction given a 3D tensor input
                ///
                template <typename ttensor3d>
                auto output(const ttensor3d& input) const
                {
                        assert(m_feature >= 0 && m_feature < input.size());
                        const auto oindex = static_cast<tensor_size_t>(input(m_feature));
                        assert(oindex >= 0 && oindex < m_outputs.size<0>());
                        return m_outputs.tensor(oindex);
                }

                ///
                /// \brief fit its parameters to the given gradients
                ///
                void fit(const task_t&, const fold_t&, const tensor4d_t& gradients, const indices_t& indices);

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
                        assert(m_outputs.size<0>() * factors.size() == m_outputs.size());
                        for (tensor_size_t r = 0, size = m_outputs.size<0>(); r < size; ++ r)
                        {
                                m_outputs.array(r) *= factors.array();
                        }
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
                auto& outputs(const tensor4d_t& outputs)
                {
                        m_outputs = outputs;
                        return *this;
                }

                ///
                /// \brief access functions
                ///
                auto feature() const { return m_feature; }
                auto fvalues() const { return m_outputs.size<0>(); }
                const auto& outputs() const { return m_outputs; }

        private:

                scalar_t fit(const task_t&, const fold_t&, const tensor4d_t& gradients,
                        const indices_t& indices, const tensor_size_t feature);

        private:

                // attributes
                tensor_size_t   m_feature{0};   ///< index of the selected feature
                tensor4d_t      m_outputs;      ///< (maximum feature values + 1, #outputs) - predictions
        };

        template <table_type type>
        std::ostream& operator<<(std::ostream& os, const wlearner_table_t<type>& table)
        {
                return os << "table=(f=" << table.feature() << ",f=" << table.fvalues() << ")";
        }

        using wlearner_real_table_t = wlearner_table_t<table_type::real>;
        using wlearner_discrete_table_t = wlearner_table_t<table_type::discrete>;
}
