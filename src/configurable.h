#pragma once

#include "stringi.h"

namespace nano
{
        ///
        /// \brief configurable object with parameters represented as string.
        ///
        struct configurable_t
        {
                ///
                /// \brief constructor
                ///
                explicit configurable_t(const string_t& params = string_t()) :
                        m_params(params)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~configurable_t() {}

                ///
                /// \brief current params (aka parameters).
                ///
                const string_t& config() const
                {
                        return m_params;
                }

                ///
                /// \brief configure with new parameters.
                ///
                const string_t& config(const string_t& params)
                {
                        return (m_params = params);
                }

        private:

                // attributes
                string_t         m_params;
        };
}
