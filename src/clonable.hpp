#pragma once

#include "stringi.h"
#include <memory>

namespace nano
{
        ///
        /// \brief the clonable interface to be used with a manager:
        ///      ::clone()                      - clone the current object
        ///      ::clone(const string_t&)       - create a new object (with the given configuration)
        ///      ::configuration()              - parametrization
        ///
        template <typename tobject>
        class clonable_t
        {
        public:

                using trobject = std::unique_ptr<tobject>;

                ///
                /// \brief constructor
                ///
                explicit clonable_t(const string_t& configuration = string_t()) :
                        m_configuration(configuration)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~clonable_t() {}

                ///
                /// \brief create an object of the same type with the given configuration
                ///
                virtual trobject clone(const string_t& configuration) const = 0;

                trobject clone() const
                {
                        return clone(config());
                }

                ///
                /// \brief current configuration (aka parameters)
                ///
                string_t config() const
                {
                        return m_configuration;
                }

        protected:

                // attributes
                string_t         m_configuration;
        };

        #define NANO_MAKE_CLONABLE(base_class) \
                virtual trobject clone(const string_t& configuration) const override \
                { \
                        return std::make_unique<base_class>(configuration); \
                }
}
