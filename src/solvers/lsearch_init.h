#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief initial step length strategies
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
        ///

        class lsearch_unit_init_t final : public lsearch_init_t
        {
        public:

                lsearch_unit_init_t() = default;

                scalar_t get(const solver_state_t&) override;

        private:

                // attributes
                int             m_iteration{0}; ///
        };

        class lsearch_linear_init_t final : public lsearch_init_t
        {
        public:

                lsearch_linear_init_t() = default;

                scalar_t get(const solver_state_t&) override;

        private:

                // attributes
                int             m_iteration{0}; ///
                scalar_t        m_prevdg{1};    ///< previous direction dot product
        };

        class lsearch_quadratic_init_t final : public lsearch_init_t
        {
        public:

                lsearch_quadratic_init_t() = default;

                scalar_t get(const solver_state_t&) override;

        private:

                // attributes
                int             m_iteration{0}; ///
                scalar_t        m_prevf{0};     ///< previous function value
        };
}
