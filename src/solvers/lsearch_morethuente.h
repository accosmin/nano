#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief More & Thunte line-search.
        ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease", by Jorge J. More and David J. Thuente
        ///
        /// NB: this implementation ports the 'dcsrch' and the 'dcstep' Fortran routines from MINPACK-2.
        ///     see http://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/
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
