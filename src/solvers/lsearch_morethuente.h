#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief More & Thunte line-search.
        ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease", by Jorge J. More and David J. Thuente
        ///
        class lsearch_morethuente_t final : public lsearch_strategy_t
        {
        public:

                lsearch_morethuente_t(const scalar_t c1, const scalar_t c2);

                lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) final;

        private:

                // attributes
                scalar_t        m_c1;                                   ///< sufficient decrease rate
                scalar_t        m_c2;                                   ///< sufficient curvature
                int             m_max_iterations{100};                  ///<
        };
}
