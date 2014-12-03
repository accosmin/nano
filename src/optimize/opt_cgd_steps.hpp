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
                tscalar cgd_step_HS(const tstate& pstate, const tstate& cstate)
                {
                        const auto& dk = pstate.d;
                        const auto& gk = pstate.g;
                        const auto& gk1 = cstate.g;
                        const auto yk = gk1 - gk;

                        return gk1.dot(yk) / dk.dot(yk);
                }

                ///
                /// \brief CGD update parameters (Fletcher and Reeves, 1964)
                ///
                template
                <
                        typename tstate,

                        typename tscalar = typename tstate::tscalar
                >
                tscalar cgd_step_FR(const tstate& pstate, const tstate& cstate)
                {
                        const auto& gk = pstate.g;
                        const auto& gk1 = cstate.g;

                        return gk1.squaredNorm() / gk.squaredNorm();
                }

                ///
                /// \brief CGD update parameters (Polak and Ribiere, 1969)
                ///
                template
                <
                        typename tstate,

                        typename tscalar = typename tstate::tscalar
                >
                tscalar cgd_step_PR(const tstate& pstate, const tstate& cstate)
                {
                        const auto& gk = pstate.g;
                        const auto& gk1 = cstate.g;
                        const auto yk = gk1 - gk;

                        return gk1.dot(yk) / gk.squaredNorm();
                }

                ///
                /// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987)
                ///
                template
                <
                        typename tstate,

                        typename tscalar = typename tstate::tscalar
                >
                tscalar cgd_step_CD(const tstate& pstate, const tstate& cstate)
                {
                        const auto& dk = pstate.d;
                        const auto& gk = pstate.g;
                        const auto& gk1 = cstate.g;

                        return - gk1.squaredNorm() / dk.dot(gk);
                }

                ///
                /// \brief CGD update parameters (Liu and Storey, 1991)
                ///
                template
                <
                        typename tstate,

                        typename tscalar = typename tstate::tscalar
                >
                tscalar cgd_step_LS(const tstate& pstate, const tstate& cstate)
                {
                        const auto& dk = pstate.d;
                        const auto& gk = pstate.g;
                        const auto& gk1 = cstate.g;
                        const auto yk = gk1 - gk;

                        return - gk1.dot(yk) / dk.dot(gk);
                }

                ///
                /// \brief CGD update parameters (Dai and Yuan, 1999)
                ///
                template
                <
                        typename tstate,

                        typename tscalar = typename tstate::tscalar
                >
                tscalar cgd_step_DY(const tstate& pstate, const tstate& cstate)
                {
                        const auto& dk = pstate.d;
                        const auto& gk = pstate.g;
                        const auto& gk1 = cstate.g;
                        const auto yk = gk1 - gk;

                        return - gk1.squaredNorm() / dk.dot(yk);
                }

                ///
                /// \brief CGD update parameters (Hager and Zhang, 2005)
                ///
                template
                <
                        typename tstate,

                        typename tscalar = typename tstate::tscalar
                >
                tscalar cgd_step_N(const tstate& pstate, const tstate& cstate)
                {
                        const auto& dk = pstate.d;
                        const auto& gk = pstate.g;
                        const auto& gk1 = cstate.g;
                        const auto yk = gk1 - gk;
                        const tscalar div = 1 / dk.dot(yk);

                        return (yk - 2 * dk * yk.squaredNorm() * div).dot(gk1 * div);
                }
        }
}

