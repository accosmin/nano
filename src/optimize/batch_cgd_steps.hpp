#pragma once

namespace ncv
{
        namespace optimize
        {
                // these variantions have been implemented following
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
                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;

                                return gk1.dot(yk) / dk.dot(yk);
                        }

                        const char* ls_failed_message() const
                        {
                                return "line-search failed for CGD-HS!";
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
                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;

                                return gk1.squaredNorm() / gk.squaredNorm();
                        }

                        const char* ls_failed_message() const
                        {
                                return "line-search failed for CGD-FR!";
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
                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;

                                return gk1.dot(yk) / gk.squaredNorm();
                        }

                        const char* ls_failed_message() const
                        {
                                return "line-search failed for CGD-PR!";
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
                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;

                                return - gk1.squaredNorm() / dk.dot(gk);
                        }

                        const char* ls_failed_message() const
                        {
                                return "line-search failed for CGD-CD!";
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
                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;

                                return - gk1.dot(yk) / dk.dot(gk);
                        }

                        const char* ls_failed_message() const
                        {
                                return "line-search failed for CGD-LS!";
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
                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;

                                return - gk1.squaredNorm() / dk.dot(yk);
                        }

                        const char* ls_failed_message() const
                        {
                                return "line-search failed for CGD-DY!";
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
                        tscalar operator()(const tstate& pstate, const tstate& cstate) const
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;
                                const tscalar div = 1 / dk.dot(yk);

                                return (yk - 2 * dk * yk.squaredNorm() * div).dot(gk1 * div);
                        }

                        const char* ls_failed_message() const
                        {
                                return "line-search failed for CGD-N!";
                        }
                };
        }
}

