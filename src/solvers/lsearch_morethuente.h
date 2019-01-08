#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief More & Thuente line-search.
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
        ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease", by Jorge J. More and David J. Thuente
        ///
        /// NB: this implementation uses the notation and the version described in Nocedal & Wright's book.
        ///
        class lsearch_morethuente_t final : public lsearch_strategy_t
        {
        public:

                lsearch_morethuente_t(const scalar_t c1, const scalar_t c2) :
                        lsearch_strategy_t(c1, c2)
                {
                }

                lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) final;

        private:

                // attributes
                int             m_max_iterations{100};                  ///<
        };
}
