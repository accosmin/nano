#ifndef NANOCV_L2_CRITERION_H
#define NANOCV_L2_CRITERION_H

#include "criterion.h"

namespace ncv
{        
        ///
        /// \brief L2-norm regularized loss
        ///
        class l2_criterion_t : public criterion_t
        {
        public:
                
                using criterion_t::robject_t;
                
                NANOCV_MAKE_CLONABLE(l2_criterion_t)

                ///
                /// \brief constructor
                ///
                l2_criterion_t(const string_t& = string_t());
                
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

        protected:

                ///
                /// \brief update statistics with a new sample
                ///
                virtual void cumulate(const vector_t& input, const vector_t& target, const loss_t& loss);
        };
}

#endif // NANOCV_L2_CRITERION_H
