#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief
        ///
        class lsearch_consistent_init_t final : public lsearch_init_t
        {
        public:

                scalar_t get(const solver_state_t&) final;

        private:

                // attributes
                bool            m_first{true};  ///< check if first iteration
                scalar_t        m_prevf{0};     ///< previous function evaluation
                scalar_t        m_prevt0{1};    ///< previous step length
                scalar_t        m_prevdg{1};    ///< previous direction dot product
        };
}
