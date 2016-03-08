#pragma once

#include "avg_criterion.h"

namespace cortex
{
        ///
        /// \brief variational regularized loss
        /// (e.g. penalize high variance across training samples),
        ///     like in EBBoost/VadaBoost: http://www.cs.columbia.edu/~jebara/papers/vadaboost.pdf
        ///     or my PhD thesis: http://infoscience.epfl.ch/record/177356?ln=fr
        ///
        class avg_var_criterion_t : public avg_criterion_t
        {
        public:

                ZOB_MAKE_CLONABLE(avg_var_criterion_t, "variational (VadaBoost-like) regularized loss")

                ///
                /// \brief constructor
                ///
                explicit avg_var_criterion_t(const string_t& = string_t());

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

        private:

                // attributes
                scalar_t        m_value2;        ///< cumulated squared loss value
                vector_t        m_vgrad2;        ///< cumulated loss value multiplied with the gradient
        };
}
