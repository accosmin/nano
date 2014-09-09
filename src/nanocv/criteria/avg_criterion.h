#ifndef NANOCV_AVG_CRITERION_H
#define NANOCV_AVG_CRITERION_H

#include "criterion.h"

namespace ncv
{        
        ///
        /// \brief averate loss
        ///
        class avg_criterion_t : public criterion_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(avg_criterion_t)

                ///
                /// \brief constructor
                ///
                avg_criterion_t(const string_t& configuration = string_t(),
                                const string_t& description = "average loss");
                
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
                /// \brief total number of processed samples
                ///
                virtual size_t count() const;

                ///
                /// \brief check if the criterion has a regularization term to tune
                ///
                virtual bool can_regularize() const;

        protected:

                ///
                /// \brief update statistics with a new sample
                ///
                virtual void accumulate(const vector_t& output, const vector_t& target, const loss_t&, scalar_t weight);

        protected:

                // attributes
                scalar_t                m_value;        ///< cumulated loss value
                vector_t                m_vgrad;        ///< cumulated gradient
                scalar_t                m_error;        ///< cumulated loss error                
                size_t                  m_count;        ///< #processed samples
        };
}

#endif // NANOCV_AVG_CRITERION_H
