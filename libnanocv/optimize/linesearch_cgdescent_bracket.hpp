#pragma once

#include <vector>
#include "linesearch_cgdescent_updateU.hpp"

//#include <iostream>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief bracket the initial line-search step length (see CG_DESCENT)
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                std::pair<tstep, tstep> cgdescent_bracket(const tstep& step0, tstep c,
                        const tscalar epsilon,
                        const tscalar theta,
                        const tscalar ro,
                        const tsize max_iters = 16)
                {
                        const tscalar depsilon = c.phi0() + epsilon * std::fabs(c.phi0());

                        std::vector<tstep> steps;
                        for (tsize i = 0; i <= max_iters && c; i ++)
                        {
//                                std::cout << "bracket [i = " << i
//                                          << "]: c = " << c.alpha()
//                                          << ", phi = " << c.phi() << "/" << depsilon
//                                          << ", gphi = " << c.gphi() << std::endl;

                                if (c.gphi() >= 0)
                                {
                                        for (auto it = steps.rbegin(); it != steps.rend(); ++ it)
                                        {
//                                                std::cout << "bracket [i = " << i << "]: c = " << c.alpha()
//                                                          << ", it = " << it->alpha() << std::endl;

                                                if (it->phi() <= epsilon)
                                                {
                                                        return std::make_pair(*it, c);
                                                }
                                        }

                                        return std::make_pair(step0, c);
                                }

                                if (c.gphi() < 0 && c.phi() > depsilon)
                                {
                                        return cgdescent_updateU(step0, step0, c, epsilon, theta);
                                }

                                else
                                {
                                        steps.push_back(c);
                                        c.reset_with_grad(ro * c.alpha());
                                }
                        }

                        // NOK, give up
                        return std::make_pair(step0, step0);
                }
        }
}

