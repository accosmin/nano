#pragma once

#include <string>
#include <memory>

namespace ncv
{
        ///
        /// the clonable interface to be used with a manager:
        ///      ::make(const std::string&)     - create a new object (with the given configuration)
        ///      ::clone()                      - create a copy of the current object
        ///      ::configuration()              - parametrization
        ///      ::description()                - short description (configuration included)
        ///
        template
        <
                typename tobject
        >
        class clonable_t
        {        
        public:

                typedef std::shared_ptr<tobject> robject_t;

                ///
                /// \brief constructor
                ///
                clonable_t(const std::string& configuration)
                        :       m_configuration(configuration)
                {
                }

                ///
                /// \brief create an object clone
                ///
                virtual robject_t make(const std::string& configuration) const = 0;
                virtual robject_t clone() const = 0;
                
                ///
                /// \brief describe the object
                ///
                std::string configuration() const { return m_configuration; }
                virtual std::string description() const = 0;

        protected:

                // attributes
                std::string     m_configuration;
        };

        #define NANOCV_MAKE_CLONABLE(base_class, description_text) \
                virtual robject_t make(const std::string& configuration) const \
                { \
                        return robject_t(new base_class(configuration)); \
                } \
                virtual robject_t clone() const \
                { \
                        return robject_t(new base_class(*this)); \
                } \
                virtual std::string description() const \
                { \
                        return description_text; \
                }
}
