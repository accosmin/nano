#include <utility>
#include "core/cubic.h"
#include "core/quadratic.h"
#include "lsearch_interpolate.h"

using namespace nano;

///
/// \brief bisection interpolation in the [step0, step1] line-search interval
///
static auto ls_bisection(const lsearch_step_t& step0, const lsearch_step_t& step1)
{
        return (step0.alpha() + step1.alpha()) / 2;
}

///
/// \brief quadratic interpolation in the [step0, step1] line-search interval
///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.58
///
/// NB: using the gradient at step0
/// NB: the step-length at step0 may be different than zero
///
static auto ls_quadratic(const lsearch_step_t& step0, const lsearch_step_t& step1)
{
        const auto x0 = step0.alpha(), f0 = step0.phi(), g0 = step0.gphi();
        const auto x1 = step1.alpha(), f1 = step1.phi();

        auto min = std::numeric_limits<scalar_t>::infinity();

        const auto q = quadratic_t<scalar_t>{x0, f0, g0, x1, f1};
        if (q)
        {
                q.extremum(min);
        }

        return min;
}

///
/// \brief cubic interpolation in the [step0, step1] line-search interval
///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
///
static auto ls_cubic(const lsearch_step_t& step0, const lsearch_step_t& step1)
{
        const auto x0 = step0.alpha(), f0 = step0.phi(), g0 = step0.gphi();
        const auto x1 = step1.alpha(), f1 = step1.phi(), g1 = step1.gphi();

        auto min1 = std::numeric_limits<scalar_t>::infinity();
        auto min2 = std::numeric_limits<scalar_t>::infinity();

        const auto c = cubic_t<scalar_t>{x0, f0, g0, x1, f1, g1};
        if (c)
        {
                c.extremum(min1, min2);
        }

        return std::make_pair(min1, min2);
}

///
/// \brief zoom-in in the bracketed interval,
///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
///
static lsearch_step_t zoom(
        const scalar_t c1, const scalar_t c2,
        const lsearch_step_t& step0, lsearch_step_t steplo, lsearch_step_t stephi)
{
        lsearch_step_t stept(step0);

        scalar_t t;
        for (int i = 0; i < 100 && std::fabs(steplo.alpha() - stephi.alpha()) > lsearch_step_t::minimum(); i ++)
        {
                // try various interpolation methods
                const auto tb = ls_bisection(steplo, stephi);
                const auto tq = ls_quadratic(steplo, stephi);
                const auto tc = ls_cubic(steplo, stephi);

                t = tb;

                std::vector<scalar_t> trials;
                trials.push_back(tb);
                trials.push_back(tq);
                trials.push_back(tc.first);
                trials.push_back(tc.second);

                // choose the valid interpolation step closest to the minimum value step
                const scalar_t tmin = std::min(steplo.alpha(), stephi.alpha());
                const scalar_t tmax = std::max(steplo.alpha(), stephi.alpha());
                const scalar_t teps = (tmax - tmin) / 20;

                scalar_t best_dist = std::numeric_limits<scalar_t>::max();
                for (const auto tt : trials)
                {
                        if (std::isfinite(tt) && tmin + teps < tt && tt < tmax - teps)
                        {
                                const scalar_t dist = std::fabs(tt - steplo.alpha());
                                if (dist < best_dist)
                                {
                                        best_dist = dist;
                                        t = tt;
                                }
                        }
                }

                // check sufficient decrease
                if (!stept.update(t))
                {
                        return step0;
                }

                if (!stept.has_armijo(c1) || stept.phi() >= steplo.phi())
                {
                        stephi = stept;
                }

                // check curvature
                else
                {
                        if (stept.has_strong_wolfe(c2))
                        {
                                return stept;
                        }

                        if (stept.gphi() * (stephi.alpha() - steplo.alpha()) >= 0)
                        {
                                stephi = steplo;
                        }

                        steplo = stept;
                }
        }

        // NOK, give up
        return step0;
}

lsearch_interpolate_t::lsearch_interpolate_t(const scalar_t c1, const scalar_t c2) :
        m_c1(c1),
        m_c2(c2)
{
}

lsearch_step_t lsearch_interpolate_t::get(const lsearch_step_t& step0, const scalar_t t0)
{
        // previous step
        lsearch_step_t stepp = step0;

        // current step
        lsearch_step_t stept = step0;

        scalar_t t = t0;
        for (int i = 1; i < m_max_iterations && t < lsearch_step_t::maximum(); i ++)
        {
                // check sufficient decrease
                if (!stept.update(t))
                {
                        return step0;
                }

                if (!stept.has_armijo(m_c1) || (stept.func() >= stepp.func() && i > 1))
                {
                        return zoom(m_c1, m_c2, step0, stepp, stept);
                }

                // check curvature
                if (stept.has_strong_wolfe(m_c2))
                {
                        return stept;
                }

                if (stept.gphi() >= scalar_t(0))
                {
                        return zoom(m_c1, m_c2, step0, stept, stepp);
                }

                stepp = stept;
                t *= 3;
        }

        // NOK, give up
        return step0;
}
