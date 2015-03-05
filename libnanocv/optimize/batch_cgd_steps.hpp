#pragma once

#include <algorithm>

namespace ncv
{
        namespace optimize
        {
                // these variations have been implemented following
                //      "A survey of nonlinear conjugate gradient methods"
                //      by William W. Hager and Hongchao Zhang

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

                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                return  cstate.g.dot(cstate.g - pstate.g) /
                                        pstate.d.dot(cstate.g - pstate.g);
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

                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                return  cstate.g.dot(cstate.g) /
                                        pstate.g.dot(pstate.g);
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

                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                return  std::max(tscalar(0),                    // PR+
                                        cstate.g.dot(cstate.g - pstate.g) /
                                        pstate.g.dot(pstate.g));
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

                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                return -cstate.g.dot(cstate.g) /
                                        pstate.d.dot(cstate.g);
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

                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                return -cstate.g.dot(cstate.g - pstate.g) /
                                        pstate.d.dot(cstate.g);
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

                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                return  cstate.g.dot(cstate.g) /
                                        pstate.d.dot(cstate.g - pstate.g);
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

                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                const tscalar div = 1 / pstate.d.dot(pstate.g);

                                return  (cstate.g - pstate.g - 2 * pstate.d * pstate.g.dot(pstate.g) * div).dot
                                        (cstate.g * div);
                        }
                };
        }
}

