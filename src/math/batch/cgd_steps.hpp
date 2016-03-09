#pragma once

#include <algorithm>

namespace zob
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
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        struct cgd_step_HS
        {
                cgd_step_HS()
                {
                }

                tscalar operator()(const tstate& prev, const tstate& curr) const
                {
                        return  curr.g.dot(curr.g - prev.g) /
                                prev.d.dot(curr.g - prev.g);
                }
        };

        ///
        /// \brief CGD update parameters (Fletcher and Reeves, 1964 - see (1))
        ///
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        struct cgd_step_FR
        {
                cgd_step_FR()
                {
                }

                tscalar operator()(const tstate& prev, const tstate& curr) const
                {
                        return  curr.g.squaredNorm() /
                                prev.g.squaredNorm();
                }
        };

        ///
        /// \brief CGD update parameters (Polak and Ribiere, 1969 - see (1))
        ///
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        struct cgd_step_PRP
        {
                cgd_step_PRP()
                {
                }

                tscalar operator()(const tstate& prev, const tstate& curr) const
                {
                        return  std::max(tscalar(0),                    // PRP(+)
                                curr.g.dot(curr.g - prev.g) /
                                prev.g.squaredNorm());
                }
        };

        ///
        /// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987 - see (1))
        ///
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        struct cgd_step_CD
        {
                cgd_step_CD()
                {
                }

                tscalar operator()(const tstate& prev, const tstate& curr) const
                {
                        return -curr.g.squaredNorm() /
                                prev.d.dot(prev.g);
                }
        };

        ///
        /// \brief CGD update parameters (Liu and Storey, 1991 - see (1))
        ///
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        struct cgd_step_LS
        {
                cgd_step_LS()
                {
                }

                tscalar operator()(const tstate& prev, const tstate& curr) const
                {
                        return -curr.g.dot(curr.g - prev.g) /
                                prev.d.dot(prev.g);
                }
        };

        ///
        /// \brief CGD update parameters (Dai and Yuan, 1999 - see (1))
        ///
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        struct cgd_step_DY
        {
                cgd_step_DY()
                {
                }

                tscalar operator()(const tstate& prev, const tstate& curr) const
                {
                        return  curr.g.squaredNorm() /
                                prev.d.dot(curr.g - prev.g);
                }
        };

        ///
        /// \brief CGD update parameters (Hager and Zhang, 2005 - see (1)) aka CG_DESCENT
        ///
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        struct cgd_step_N
        {
                cgd_step_N()
                {
                }

                tscalar operator()(const tstate& prev, const tstate& curr) const
                {
                        const auto y = curr.g - prev.g;
                        const tscalar div = +1 / prev.d.dot(y);

                        const tscalar pd2 = prev.d.template lpNorm<2>();
                        const tscalar pg2 = prev.g.template lpNorm<2>();
                        const tscalar eta = -1 / (pd2 * std::min(tscalar(0.01), pg2));

                        // N+ (see modification in
                        //      "A NEW CONJUGATE GRADIENT METHOD WITH GUARANTEED DESCENT AND AN EFFICIENT LINE SEARCH")
                        return  std::max(eta,
                                         div * (y - 2 * prev.d * y.squaredNorm() * div).dot(curr.g));
                }
        };

        ///
        /// \brief CGD update parameters (Dai and Yuan, 2001  - see (2), page 21)
        ///
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        struct cgd_step_DYHS
        {
                cgd_step_DYHS()
                {
                }

                tscalar operator()(const tstate& prev, const tstate& curr) const
                {
                        const tscalar dy = cgd_step_DY<tstate>()(prev, curr);
                        const tscalar hs = cgd_step_HS<tstate>()(prev, curr);

                        return std::max(tscalar(0), std::min(dy, hs));
                }
        };

        ///
        /// \brief CGD update parameters (Dai, 2002 - see (2), page 22)
        ///
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        struct cgd_step_DYCD
        {
                cgd_step_DYCD()
                {
                }

                tscalar operator()(const tstate& prev, const tstate& curr) const
                {
                        return  curr.g.squaredNorm() /
                                std::max(prev.d.dot(curr.g - prev.g), -prev.d.dot(prev.g));
                }
        };
}


