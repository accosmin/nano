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
                
                NANOCV_MAKE_CLONABLE(avg_l2_criterion_t)

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
                /// \brief average loss value
                ///
                virtual scalar_t value() const;

                ///
                /// \brief average error value
                ///
                virtual scalar_t error() const;

                ///
                /// \brief average gradient
                ///
                virtual vector_t vgrad() const;

                ///
                /// \brief check if the criterion has a regularization term to tune
                ///
                virtual bool can_regularize() const;

        protected:

                ///
                /// \brief update statistics with a new sample
                ///
                virtual void accumulate(const vector_t& input, const vector_t& target, const loss_t&, scalar_t weight);
        };
}
