#pragma once

#include "stringi.h"

namespace nano
{
        ///
        /// \brief the clonable interface to be used with a manager:
        ///      ::config()     - current parameters
        ///
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
