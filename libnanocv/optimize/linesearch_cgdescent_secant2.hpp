#pragma once

#include "linesearch_cgdescent_secant.hpp"
#include "linesearch_cgdescent_update.hpp"
#include <limits>

#include <iostream>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief [a, b] line-search interval double secant update (see CG_DESCENT)
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                std::pair<tstep, tstep> cgdescent_secant2(const tstep& a, const tstep& b,
                        const tscalar epsilon, const tscalar theta)
                {
                        // (Hager & Zhang 2005, p. 15)
                        const tstep c = cgdescent_secant(a, b);

//                        std::cout << "secant (" << a.alpha() << ", " << b.alpha()
//                                  << ") = " << c.alpha() << std::endl;

                        tstep A(a), B(b);
                        std::tie(A, B) = cgdescent_update(a, b, c, epsilon, theta);

//                        std::cout << "update (" << a.alpha() << ", " << b.alpha() << ", " << c.alpha()
//                                  << ") = (" << A.alpha() << ", " << B.alpha() << ")" << std::endl;

                        if (std::fabs(c.alpha() - A.alpha()) < std::numeric_limits<tscalar>::epsilon())
                        {
//                                std::cout << "secant (a, A) [" << a.alpha() << ", " << A.alpha()
//                                          << "] = " << cgdescent_secant(a, A).alpha() << std::endl;

                                return cgdescent_update(A, B, cgdescent_secant(a, A), epsilon, theta);
                        }

                        else if (std::fabs(c.alpha() - B.alpha()) < std::numeric_limits<tscalar>::epsilon())
                        {
//                                std::cout << "secant (b, B) [" << b.alpha() << ", " << B.alpha()
//                                          << "] = " << cgdescent_secant(b, B).alpha() << std::endl;

                                return cgdescent_update(A, B, cgdescent_secant(b, B), epsilon, theta);
                        }

                        else
                        {
                                return std::make_pair(A, B);
                        }
                }
        }
}

