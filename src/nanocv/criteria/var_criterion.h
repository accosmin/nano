#ifndef NANOCV_VAR_CRITERION_H
#define NANOCV_VAR_CRITERION_H

#include "criterion.h"

namespace ncv
{        
        ///
        /// \brief variational regularized loss
        /// (e.g. penalize high variance across training samples),
        ///     like in EBBoost/VadaBoost: http://www.cs.columbia.edu/~jebara/papers/vadaboost.pdf
        ///     or my PhD thesis: http://infoscience.epfl.ch/record/177356?ln=fr
        ///
        class var_criterion_t : public criterion_t
        {
        public:

                ///
                /// \brief constructor
                ///
                var_criterion_t(const string_t& = string_t());
                
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

#endif // NANOCV_VAR_CRITERION_H
