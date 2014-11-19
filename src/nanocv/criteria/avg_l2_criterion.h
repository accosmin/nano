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
                avg_l2_criterion_t(const string_t& = string_t());
                
                ///
                /// \brief reset statistics and settings
                ///
                virtual void reset();

                ///
                /// \brief cumulate statistics
                ///
                virtual criterion_t& operator+=(const criterion_t&);

                ///
                /// \brief cumulated loss value
                ///
                virtual scalar_t value() const;

                ///
                /// \brief cumulated error value
                ///
                virtual scalar_t error() const;

                ///
                /// \brief cumulated gradient
                ///
                virtual vector_t vgrad() const;

                ///
                /// \brief check if the criterion has a regularization term to tune
                ///
                virtual bool can_regularize() const;

        protected:

                ///
                /// \brief update statistics with the loss value/error/gradient for a sample
                ///
                virtual void accumulate(scalar_t value, scalar_t error);
                virtual void accumulate(const vector_t& vgrad, scalar_t value, scalar_t error);
        };
}
