#pragma once

#include "criterion.h"

namespace ncv
{        
        ///
        /// \brief average loss
        ///
        class avg_criterion_t : public criterion_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(avg_criterion_t, "average loss")

                ///
                /// \brief constructor
                ///
                avg_criterion_t(const string_t& configuration = string_t());
                
                ///
                /// \brief destructor
                ///
                virtual ~avg_criterion_t() {}
                                
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
                /// \brief total number of processed samples
                ///
                virtual size_t count() const;

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

        protected:

                // attributes
                scalar_t                m_value;        ///< cumulated loss value
                vector_t                m_vgrad;        ///< cumulated gradient
                scalar_t                m_error;        ///< cumulated loss error                
                size_t                  m_count;        ///< #processed samples
        };
}

