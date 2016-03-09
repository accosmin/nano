#pragma once

#include "avg_criterion.h"

namespace zob
{        
        ///
        /// \brief L2-norm regularized loss
        ///
        class avg_l2_criterion_t : public avg_criterion_t
        {
        public:
                
                ZOB_MAKE_CLONABLE(avg_l2_criterion_t, "L2-norm regularized loss")

                ///
                /// \brief constructor
                ///
                explicit avg_l2_criterion_t(const string_t& = string_t());

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
        };
}
