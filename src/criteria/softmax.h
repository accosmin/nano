#pragma once

#include "criterion.h"

namespace nano
{
        ///
        /// \brief average loss
        ///
        class softmax_criterion_t : public criterion_t
        {
        public:

                NANO_MAKE_CLONABLE(softmax_criterion_t, "beta=5[1,10]")

                ///
                /// \brief constructor
                ///
                explicit softmax_criterion_t(const string_t& configuration = string_t());

                ///
                /// \brief destructor
                ///
                virtual ~softmax_criterion_t() {}

                ///
                /// \brief cumulated loss value
                ///
                virtual scalar_t value() const override;

                ///
                /// \brief cumulated gradient
                ///
                virtual vector_t vgrad() const override;

                ///
                /// \brief check if the criterion has a regularization term to tune
                ///
                virtual bool can_regularize() const override;

        protected:

                ///
                /// \brief reset statistics
                ///
                virtual void clear() override;

                ///
                /// \brief update statistics with the loss value/error/gradient for a sample
                ///
                virtual void accumulate(scalar_t value) override;
                virtual void accumulate(const vector_t& vgrad, scalar_t value) override;

                ///
                /// \brief update statistics with cumulated samples
                ///
                virtual void accumulate(const criterion_t& other) override;

        protected:

                // attributes
                scalar_t                m_beta;
                scalar_t                m_value;        ///< cumulated loss value
                vector_t                m_vgrad;        ///< cumulated gradient
        };
}

