#ifndef NANOCV_OPTIMIZE_STATE_HPP
#define NANOCV_OPTIMIZE_STATE_HPP

#include "core/optimize/problem.hpp"

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////////
                // optimization state:
                //      current point (x), function value (f), gradient (g),
                //      descent direction (d) & line-search step (t).
                ////////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar,
                        typename tsize
                >
                struct state_t
                {
                        typedef problem_t<tscalar, tsize>               tproblem;
                        typedef typename tproblem::vector_t             tvector;

                        // constructor
                        state_t(tsize size = 0)
                                :       x(size), g(size), d(size),
                                        f(std::numeric_limits<tscalar>::max()),
                                        t(1.0)
                        {
                        }

                        // constructor
                        state_t(const tproblem& problem, const tvector& x0)
                                :       state_t(problem.size())
                        {
                                x = x0;
                                f = problem.f(x, g);
                        }

                        // update current point
                        void update(const tproblem& problem, tscalar t)
                        {
                                x.noalias() += t * d;
                                f = problem.f(x, g);
                        }

                        void update(tscalar t, tscalar ft, const tvector& gt)
                        {
                                x.noalias() += t * d;
                                f = ft;
                                g = gt;
                        }

                        // check convergence: the gradient is relatively small
                        bool converged(tscalar epsilon) const
                        {
                                return (g.template lpNorm<Eigen::Infinity>()) < epsilon * (1.0 + std::fabs(f));
                        }

                        // attributes
                        tvector         x, g, d;
                        tscalar         f, t;
                };

                // compare two optimization states
                template
                <
                        typename tscalar,
                        typename tsize
                >
                bool operator<(const state_t<tscalar, tsize>& one, const state_t<tscalar, tsize>& other)
                {
                        return one.f < other.f;
                }
        }
}

#endif // NANOCV_OPTIMIZE_STATE_HPP
