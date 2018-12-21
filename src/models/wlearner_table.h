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
        class wlearner_table_t : public wlearner_t
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
                /// \brief fit its parameters to the given gradients and feature
                ///
                scalar_t fit(const task_t&, const fold_t&, const tensor4d_t& gradients, const indices_t&,
                        const tensor_size_t feature);

                ///
                /// \brief scale the outputs by the given factor
                ///
                void scale(const scalar_t factor)
                {
                        wlearner_t::scale(m_outputs, factor);
                }

                ///
                /// \brief scale the outputs by the given factors
                ///
                void scale(const vector_t& factors)
                {
                        for (tensor_size_t r = 0, size = m_outputs.size<0>(); r < size; ++ r)
                        {
                                wlearner_t::scale(m_outputs.tensor(r), factors);
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
                auto& outputs(const tensor4d_t& outputs)
                {
                        m_outputs = outputs;
                        return *this;
                }

                ///
                /// \brief access functions
                ///
                auto fvalues() const { return m_outputs.size<0>(); }
                const auto& outputs() const { return m_outputs; }

        private:

                // attributes
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
