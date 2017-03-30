#pragma once

#include "criterion.h"

namespace nano
{
        ///
        /// \brief softmax the loss value across samples:
        ///     C(X) = softmax(L(x_i), x_i \in X; beta),
        //      C(X) = log(sum(exp(beta * L(x_i)), x_i \in X)) / beta.
        ///
        class softmax_criterion_t : public criterion_t
        {
        public:

                explicit softmax_criterion_t(const string_t& configuration = string_t());

                virtual rcriterion_t clone() const override;

                virtual scalar_t value() const override;
                virtual vector_t vgrad() const override;
                virtual bool can_regularize() const override;

        protected:

                virtual void clear() override;
                virtual void accumulate(const scalar_t value) override;
                virtual void accumulate(const vector_t& vgrad, const scalar_t value) override;
                virtual void accumulate(const criterion_t& other) override;

        protected:

                // attributes
                scalar_t                m_beta;
                scalar_t                m_value;        ///< cumulated loss value
                vector_t                m_vgrad;        ///< cumulated gradient
        };
}
