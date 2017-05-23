#pragma once

#include "text/config.h"

namespace nano
{
        ///
        /// \brief configurable object with parameters represented as a string.
        ///
        struct configurable_t
        {
                ///
                /// \brief constructor
                ///
                explicit configurable_t(const string_t& config = string_t()) :
                        m_config(config) {}

                ///
                /// \brief destructor
                ///
                virtual ~configurable_t() {}

                ///
                /// \brief current config (aka parameters).
                ///
                const string_t& config() const
                {
                        return m_config;
                }

                ///
                /// \brief configure with new parameters.
                ///
                const string_t& config(const string_t& config)
                {
                        return (m_config = config);
                }

        private:

                // attributes
                string_t         m_config;
        };
}
