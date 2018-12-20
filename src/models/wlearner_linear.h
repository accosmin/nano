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
        /// \brief a linear weak learner is performing an affine transformation of the selected feature feature:
        ///     linear(x) = a * x(feature) + b
        ///
        class NANO_PUBLIC wlearner_linear_t : public wlearner_t
        {
        public:

                ///
                /// \brief default constructor
                ///
                wlearner_linear_t() = default;

                ///
                /// \brief compute the output/prediction given a 3D tensor input
                ///
                template <typename ttensor3d>
                auto output(const ttensor3d& input) const
                {
                        assert(m_a.dims() == m_b.dims());
                        assert(m_feature >= 0 && m_feature < input.size());
                        tensor3d_t output(m_a.dims());
                        output.array() = m_a.array() * input(m_feature) + m_b.array();
                        return output;
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
                        m_a.array() *= factor;
                        m_b.array() *= factor;
                }

                ///
                /// \brief scale the outputs by the given factors
                ///
                void scale(const vector_t& factors)
                {
                        assert(factors.minCoeff() >= 0);
                        assert(factors.size() == m_a.size());
                        assert(factors.size() == m_b.size());
                        m_a.array() *= factors.array();
                        m_b.array() *= factors.array();
                }

                ///
                /// \brief serialize to disk
                ///
                bool load(ibstream_t&);
                bool save(obstream_t&) const;

                ///
                /// \brief change parameters
                ///
                auto& a(const tensor3d_t& a)
                {
                        m_a = a;
                        return *this;
                }
                auto& b(const tensor3d_t& b)
                {
                        m_b = b;
                        return *this;
                }

                ///
                /// \brief access functions
                ///
                const auto& a() const { return m_a; }
                const auto& b() const { return m_b; }

        private:

                // attributes
                tensor3d_t      m_a;            ///<
                tensor3d_t      m_b;            ///<
        };

        inline std::ostream& operator<<(std::ostream& os, const wlearner_linear_t& linear)
        {
                return os << "linear=(f=" << linear.feature() << ")";
        }
}
