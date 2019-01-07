#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief
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

        ///
        /// \brief
        ///
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

        ///
        /// \brief
        ///
        class lsearch_consistent_init_t final : public lsearch_init_t
        {
        public:

                lsearch_consistent_init_t() = default;

                scalar_t get(const solver_state_t&) override;

        private:

                // attributes
                int             m_iteration{0}; ///
                scalar_t        m_prevf{0};     ///< previous function value
                scalar_t        m_prevt{1};     ///< previous step length
                scalar_t        m_prevdg{1};    ///< previous direction dot product
        };
}
