#pragma once

#include "cortex/criterion.h"

namespace cortex
{
        ///
        /// \brief average loss
        ///
        class avg_criterion_t : public criterion_t
        {
        public:

                ZOB_MAKE_CLONABLE(avg_criterion_t, "average loss")

                ///
                /// \brief constructor
                ///
                explicit avg_criterion_t(const string_t& configuration = string_t());

                ///
                /// \brief destructor
                ///
                virtual ~avg_criterion_t() {}

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
                scalar_t                m_value;        ///< cumulated loss value
                vector_t                m_vgrad;        ///< cumulated gradient
        };
}

