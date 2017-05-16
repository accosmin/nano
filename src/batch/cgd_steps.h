#pragma once

#include "function_state.h"
#include <algorithm>

namespace nano
{
        // these variations have been implemented following:
        //      (1) "A survey of nonlinear conjugate gradient methods"
        //      by William W. Hager and Hongchao Zhang
        //
        // and
        //      (2) "Nonlinear Conjugate Gradient Methods"
        //      by Yu-Hong Dai

        ///
        /// \brief CGD update parameters (Hestenes and Stiefel, 1952 - see (1))
        ///
        struct NANO_PUBLIC cgd_step_HS
        {
                static const char* name()
                {
                        return "hs";
                }

                scalar_t operator()(const function_state_t& prev, const function_state_t& curr) const
                {
                        return  curr.g.dot(curr.g - prev.g) /
                                prev.d.dot(curr.g - prev.g);
                }
        };

        ///
        /// \brief CGD update parameters (Fletcher and Reeves, 1964 - see (1))
        ///
        struct NANO_PUBLIC cgd_step_FR
        {
                static const char* name()
                {
                        return "fr";
                }

                scalar_t operator()(const function_state_t& prev, const function_state_t& curr) const
                {
                        return  curr.g.squaredNorm() /
                                prev.g.squaredNorm();
                }
        };

        ///
        /// \brief CGD update parameters (Polak and Ribiere, 1969 - see (1))
        ///
        struct NANO_PUBLIC cgd_step_PRP
        {
                static const char* name()
                {
                        return "prp";
                }

                scalar_t operator()(const function_state_t& prev, const function_state_t& curr) const
                {
                        return  std::max(scalar_t(0),                    // PRP(+)
                                curr.g.dot(curr.g - prev.g) /
                                prev.g.squaredNorm());
                }
        };

        ///
        /// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987 - see (1))
        ///
        struct NANO_PUBLIC cgd_step_CD
        {
                static const char* name()
                {
                        return "cd";
                }

                scalar_t operator()(const function_state_t& prev, const function_state_t& curr) const
                {
                        return -curr.g.squaredNorm() /
                                prev.d.dot(prev.g);
                }
        };

        ///
        /// \brief CGD update parameters (Liu and Storey, 1991 - see (1))
        ///
        struct NANO_PUBLIC cgd_step_LS
        {
                static const char* name()
                {
                        return "ls";
                }

                scalar_t operator()(const function_state_t& prev, const function_state_t& curr) const
                {
                        return -curr.g.dot(curr.g - prev.g) /
                                prev.d.dot(prev.g);
                }
        };

        ///
        /// \brief CGD update parameters (Dai and Yuan, 1999 - see (1))
        ///
        struct NANO_PUBLIC cgd_step_DY
        {
                static const char* name()
                {
                        return "dy";
                }

                scalar_t operator()(const function_state_t& prev, const function_state_t& curr) const
                {
                        return  curr.g.squaredNorm() /
                                prev.d.dot(curr.g - prev.g);
                }
        };

        ///
        /// \brief CGD update parameters (Hager and Zhang, 2005 - see (1)) aka CG_DESCENT
        ///
        struct NANO_PUBLIC cgd_step_N
        {
                static const char* name()
                {
                        return "n";
                }

                scalar_t operator()(const function_state_t& prev, const function_state_t& curr) const
                {
                        const auto y = curr.g - prev.g;
                        const scalar_t div = +1 / prev.d.dot(y);

                        const scalar_t pd2 = prev.d.lpNorm<2>();
                        const scalar_t pg2 = prev.g.lpNorm<2>();
                        const scalar_t eta = -1 / (pd2 * std::min(scalar_t(0.01), pg2));

                        // N+ (see modification in
                        //      "A NEW CONJUGATE GRADIENT METHOD WITH GUARANTEED DESCENT AND AN EFFICIENT LINE SEARCH")
                        return  std::max(eta,
                                         div * (y - 2 * prev.d * y.squaredNorm() * div).dot(curr.g));
                }
        };

        ///
        /// \brief CGD update parameters (Dai and Yuan, 2001  - see (2), page 21)
        ///
        struct NANO_PUBLIC cgd_step_DYHS
        {
                static const char* name()
                {
                        return "dyhs";
                }

                scalar_t operator()(const function_state_t& prev, const function_state_t& curr) const
                {
                        const scalar_t dy = cgd_step_DY()(prev, curr);
                        const scalar_t hs = cgd_step_HS()(prev, curr);

                        return std::max(scalar_t(0), std::min(dy, hs));
                }
        };

        ///
        /// \brief CGD update parameters (Dai, 2002 - see (2), page 22)
        ///
        struct NANO_PUBLIC cgd_step_DYCD
        {
                static const char* name()
                {
                        return "dycd";
                }

                scalar_t operator()(const function_state_t& prev, const function_state_t& curr) const
                {
                        return  curr.g.squaredNorm() /
                                std::max(prev.d.dot(curr.g - prev.g), -prev.d.dot(prev.g));
                }
        };
}
