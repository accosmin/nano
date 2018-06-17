#pragma once

#include "lsearch_step.h"

namespace nano
{
        ///
        /// \brief compute the step length of the line search procedure.
        ///
        class lsearch_length_t
        {
        public:
                virtual ~lsearch_length_t() = default;
                virtual lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) = 0;
        };

        class lsearch_cgdescent_length_t final : public lsearch_length_t
        {
        public:
                lsearch_cgdescent_length_t(const scalar_t c1, const scalar_t c2);

                lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) final;

        private:

                // attributes
                scalar_t        m_c1;           ///< sufficient decrease rate
                scalar_t        m_c2;           ///< sufficient curvature
        };

        class lsearch_interpolation_length_t final : public lsearch_length_t
        {
        public:
                lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) final;

        private:

                // attributes
                scalar_t        m_c1;           ///< sufficient decrease rate
                scalar_t        m_c2;           ///< sufficient curvature
        };

        class lsearch_backtrack_armijo_length_t final : public lsearch_length_t
        {
        public:
                lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) final;

        private:

                // attributes
                scalar_t        m_c1;           ///< sufficient decrease rate
                scalar_t        m_c2;           ///< sufficient curvature
        };

        class lsearch_backtrack_wolfe_length_t final : public lsearch_length_t
        {
        public:
                lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) final;

        private:

                // attributes
                scalar_t        m_c1;           ///< sufficient decrease rate
                scalar_t        m_c2;           ///< sufficient curvature
        };

        class lsearch_backtrack_strong_wolfe_length_t final : public lsearch_length_t
        {
        public:
                lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) final;

        private:

                // attributes
                scalar_t        m_c1;           ///< sufficient decrease rate
                scalar_t        m_c2;           ///< sufficient curvature
        };
}
