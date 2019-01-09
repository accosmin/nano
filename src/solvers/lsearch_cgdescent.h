#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief CG_DESCENT:
        ///     see "A new conjugate gradient method with guaranteed descent and an efficient line search",
        ///     by William W. Hager & HongChao Zhang, 2005
        ///
        ///     see "Algorithm 851: CG_DESCENT, a Conjugate Gradient Method with Guaranteed Descent",
        ///     by William W. Hager & HongChao Zhang, 2006
        ///
        class lsearch_cgdescent_t final : public lsearch_strategy_t
        {
        public:

                lsearch_cgdescent_t(const scalar_t c1, const scalar_t c2) :
                        lsearch_strategy_t(c1, c2)
                {
                }

                lsearch_step_t get(const lsearch_step_t& step0, const scalar_t t0) final;

        private:

                // attributes
                scalar_t        m_epsilon{static_cast<scalar_t>(1e-6)}; ///<
                scalar_t        m_theta{static_cast<scalar_t>(0.5)};    ///<
                scalar_t        m_gamma{static_cast<scalar_t>(0.66)};   ///<
                scalar_t        m_delta{static_cast<scalar_t>(0.7)};    ///<
                scalar_t        m_omega{static_cast<scalar_t>(1e-3)};   ///<
                scalar_t        m_ro{static_cast<scalar_t>(5.0)};       ///<
                scalar_t        m_sumQ{0};                              ///<
                scalar_t        m_sumC{0};                              ///<
                int             m_max_iterations{100};                  ///<
        };
}
