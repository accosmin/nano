#pragma once

#include "avg_criterion.h"

namespace ncv
{        
        ///
        /// \brief L2-norm regularized loss
        ///
        class avg_l2_criterion_t : public avg_criterion_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(avg_l2_criterion_t, "L2-norm regularized loss")

                ///
                /// \brief constructor
                ///
                explicit avg_l2_criterion_t(const string_t& = string_t());
                
                ///
                /// \brief reset statistics and settings
                ///
                virtual void reset() override;

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
                /// \brief update statistics with the loss value/error/gradient for a sample
                ///
                virtual void accumulate(scalar_t value) override;
                virtual void accumulate(const vector_t& vgrad, scalar_t value) override;

                ///
                /// \brief update statistics with cumulated samples
                ///
                virtual void accumulate(const criterion_t& other) override;
        };
}
