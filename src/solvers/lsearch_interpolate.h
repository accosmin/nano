#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief interpolation-based line-search (More & Thuente -like?!),
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
        ///
        class lsearch_interpolate_t final : public lsearch_strategy_t
        {
        public:

                lsearch_interpolate_t(const scalar_t c1, const scalar_t c2);

                lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) final;

        private:

                // attributes
                scalar_t        m_c1;                                   ///< sufficient decrease rate
                scalar_t        m_c2;                                   ///< sufficient curvature
                int             m_max_iterations{100};                  ///<
        };
}
