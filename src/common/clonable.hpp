#ifndef NANOCV_CLONABLE_H
#define NANOCV_CLONABLE_H

#include <string>
#include <memory>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // the clonable interface to be used with a manager:
        //      ::clone(const std::string&)     - create a new object (with the given parameters)
        //      ::description()                 - short description (parameters included)
        // hint: use register_object<base, derived> to register objects to the manager.
        /////////////////////////////////////////////////////////////////////////////////////////
        
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
                virtual robject_t clone(const std::string& params) const = 0;
                
                // describe the object
                virtual std::string description() const = 0;
        };

        // implements the clonable_t interface
        #define NCV_MAKE_CLONABLE(object_class, base_class, description_str) \
                typedef typename clonable_t<base_class>::robject_t robject_t; \
                \
                virtual robject_t clone(const std::string& params) const \
                { \
                        return std::make_shared<object_class>(params); \
                } \
                virtual robject_t clone() const \
                { \
                        return std::make_shared<object_class>(*this); \
                } \
                \
                virtual std::string description() const { return #description_str; }
}

#endif // NANOCV_CLONABLE_H

