#ifndef NANOCV_LOSS_HINGE_H
#define NANOCV_LOSS_HINGE_H

#include "ncv_loss.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // hinge loss.
        ////////////////////////////////////////////////////////////////////////////////
        
        class hinge_loss_t : public loss_t
        {
        public:
                
                // create an object clone
                virtual rloss_t clone(const string_t& /*params*/) const
                {
                        return rloss_t(new hinge_loss_t(*this));
                }

                // describe the object
                virtual const char* name() const { return "hinge"; }
                virtual const char* desc() const { return "hinge loss"; }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return eclass_error(targets, scores);
                }

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual scalar_t vgrad(const vector_t& targets, const vector_t& scores, vector_t& grads) const;
        };
}

#endif // NANOCV_LOSS_HINGE_H
