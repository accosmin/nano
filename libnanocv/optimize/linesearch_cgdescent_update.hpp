#pragma once

#include <utility>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief [a, b] line-search interval update (see CG_DESCENT)
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                std::pair<tstep, tstep> cgdescent_update(tstep a, tstep b, tstep c,
                        const tscalar epsilon, const tscalar theta,
                        const tsize max_iters = 64)
                {
                        const tscalar depsilon = a.phi0() + epsilon * std::fabs(a.phi0());

                        // (Hager & Zhang 2005, p. 13)
                        if (c.alpha() <= a.alpha() || c.alpha() >= b.alpha())
                        {
                                return std::make_pair(a, b);
                        }

                        else if (c.gphi() >= 0)
                        {
                                return std::make_pair(a, c);
                        }

                        else if (c.phi() <= depsilon)
                        {
                                return std::make_pair(c, b);
                        }

                        else
                        {
                                b = c;

                                // NB: we are using <c> as the <d> from the original paper!
                                for (size_t i = 1; i <= max_iters; i ++)
                                {
                                        c.reset_with_grad((1 - theta) * a.alpha() + theta * b.alpha());

                                        if (c.gphi() >= 0)
                                        {
                                                return std::make_pair(a, c);
                                        }

                                        else if (c.phi() <= depsilon)
                                        {
                                                a = c;
                                        }

                                        else
                                        {
                                                b = c;
                                        }
                                }
                        }

                        // OK, give up
                        return std::make_pair(a, b);
                }
        }
}

