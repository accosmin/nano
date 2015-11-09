#pragma once

#include <string>
#include <memory>

namespace cortex
{
        ///
        /// \brief the clonable interface to be used with a manager:
        ///      ::clone()                      - clone the current object
        ///      ::clone(const tstring&)        - create a new object (with the given configuration)
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

                using trobject = std::shared_ptr<tobject>;
                using tstring = std::string;

                ///
                /// \brief constructor
                ///
                clonable_t(const tstring& configuration)
                        :       m_configuration(configuration)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~clonable_t() {}

                ///
                /// \brief create an object clone
                ///
                virtual trobject clone(const tstring& configuration) const = 0;
                virtual trobject clone() const = 0;

                ///
                /// \brief short description (e.g. purpose, parameters)
                ///
                virtual tstring description() const = 0;

                ///
                /// \brief current configuration (aka parameters)
                ///
                tstring configuration() const { return m_configuration; }

        protected:

                // attributes
                tstring         m_configuration;
        };

        #define NANOCV_MAKE_CLONABLE(base_class, description_text) \
                virtual trobject clone(const tstring& configuration) const override \
                { \
                        return std::make_shared<base_class>(configuration); \
                } \
                virtual trobject clone() const override \
                { \
                        return std::make_shared<base_class>(*this); \
                } \
                virtual tstring description() const override \
                { \
                        return description_text; \
                }
}
