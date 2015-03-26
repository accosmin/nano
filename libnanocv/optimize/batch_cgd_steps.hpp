#pragma once

#include <algorithm>

namespace ncv
{
        namespace optimize
        {
                // these variations have been implemented following:
                //      "A survey of nonlinear conjugate gradient methods"
                //      by William W. Hager and Hongchao Zhang
                //
                // and
                //      "Nonlinear Conjugate Gradient Methods"
                //      by Yu-Hong Dai

                ///
                /// \brief CGD update parameters (Hestenes and Stiefel, 1952)
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
                /// \brief CGD update parameters (Fletcher and Reeves, 1964)
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
                /// \brief CGD update parameters (Polak and Ribiere, 1969)
                ///
                template
                <
                        typename tstate,
                        typename tscalar = typename tstate::tscalar
                >
                struct cgd_step_PR
                {
                        cgd_step_PR()
                        {
                        }

                        tscalar operator()(const tstate& prev, const tstate& curr) const
                        {
                                return  std::max(tscalar(0),                    // PR+
                                        curr.g.dot(curr.g - prev.g) /
                                        prev.g.squaredNorm());
                        }
                };

                ///
                /// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987)
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
                /// \brief CGD update parameters (Liu and Storey, 1991)
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
                /// \brief CGD update parameters (Dai and Yuan, 1999)
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
                /// \brief CGD update parameters (Hager and Zhang, 2005)
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
                                const tscalar div = 1 / prev.d.dot(curr.g - prev.g);

                                return  (curr.g - prev.g - 2 * prev.d * (curr.g - prev.g).squaredNorm() * div).dot
                                        (curr.g * div);
                        }
                };

                ///
                /// \brief CGD update parameters (Dai and Yuan, 2001, page 21 - DY + HS)
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
        }
}

