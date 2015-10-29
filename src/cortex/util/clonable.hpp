#pragma once

#include <string>
#include <memory>

namespace cortex
{
        ///
        /// \brief the clonable interface to be used with a manager:
        ///      ::make(const tstring&)         - create a new object (with the given configuration)
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
                /// \brief create an object clone
                ///
                virtual trobject make(const tstring& configuration) const = 0;
                virtual trobject clone() const = 0;

                ///
                /// \brief describe the object
                ///
                tstring configuration() const { return m_configuration; }
                virtual tstring description() const = 0;

        protected:

                // attributes
                tstring         m_configuration;
        };

        #define NANOCV_MAKE_CLONABLE(base_class, description_text) \
                virtual trobject make(const tstring& configuration) const override \
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
