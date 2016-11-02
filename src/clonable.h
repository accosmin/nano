#pragma once

#include "stringi.h"

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
                /// \brief current configuration (aka parameters).
                ///
                const string_t& config() const
                {
                        return m_configuration;
                }

        protected:

                // attributes
                string_t         m_configuration;
        };
}
