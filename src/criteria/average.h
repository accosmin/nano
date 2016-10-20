#pragma once

#include "criterion.h"

namespace nano
{
        ///
        /// \brief average loss
        ///
        class average_criterion_t : public criterion_t
        {
        public:

                explicit average_criterion_t(const string_t& configuration = string_t());

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
                scalar_t                m_value;        ///< cumulated loss value
                vector_t                m_vgrad;        ///< cumulated gradient
        };
}

