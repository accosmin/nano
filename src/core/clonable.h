#ifndef NANOCV_CLONABLE_H
#define NANOCV_CLONABLE_H

#include "string.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // the clonable interface to be used with a manager:
        //      ::clone(const string_t&)        - create a new object (with the given parameters)
        //      ::description()                 - short description (parameters included)
        // hint: use register_object<base, derived> to register objects to the manager.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        template
        <
                typename tobject
        >
        class clonable_t
        {        
        public:

                typedef std::shared_ptr<tobject> robject_t;

                // create an object clone
                virtual robject_t clone() const = 0;
                virtual robject_t clone(const string_t& params) const = 0;
                
                // describe the object
                virtual string_t description() const = 0;
        };

        // implements the clonable_t interface
        #define NCV_MAKE_CLONABLE(object_class, base_class, description_str) \
                typedef typename clonable_t<base_class>::robject_t robject_t; \
                \
                virtual robject_t clone(const string_t& params) const \
                { \
                        return robject_t(new object_class(params)); \
                } \
                virtual robject_t clone() const \
                { \
                        return robject_t(new object_class(*this)); \
                } \
                \
                virtual string_t description() const { return #description_str; }
}

#endif // NANOCV_CLONABLE_H

