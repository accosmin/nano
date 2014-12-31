#pragma once

#include "avg_criterion.h"

namespace ncv
{        
        ///
        /// \brief penalize highly the cumulated loss if the L2-norm of (group) parameters is greater than 1
        ///
        class avg_l2_barrier_criterion_t : public avg_criterion_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(avg_l2_barrier_criterion_t, "L2-norm barrier loss")

                ///
                /// \brief constructor
                ///
                avg_l2_barrier_criterion_t(const string_t& = string_t());
                
                ///
                /// \brief reset statistics and settings
                ///
                virtual void reset();

                ///
                /// \brief cumulated loss value
                ///
                virtual scalar_t value() const;

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

                ///
                /// \brief update statistics with cumulated samples
                ///
                virtual void accumulate(const criterion_t& other);
        };
}
