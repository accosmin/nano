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
                bool            m_first{true};  ///< check if first iteration
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
                bool            m_first{true};  ///< check if first iteration
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
                bool            m_first{true};  ///< check if first iteration
                scalar_t        m_prevf{0};     ///< previous function value
                scalar_t        m_prevt{1};     ///< previous step length
                scalar_t        m_prevdg{1};    ///< previous direction dot product
        };
}
