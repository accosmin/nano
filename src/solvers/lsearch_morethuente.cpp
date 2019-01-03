#include <utility>
#include "core/cubic.h"
#include "core/quadratic.h"
#include "lsearch_morethuente.h"

using namespace nano;

static auto cubic(
        const scalar_t u, const scalar_t fu, const scalar_t gu,
        const scalar_t v, const scalar_t fv, const scalar_t gv)
{
        // fit cubic: q(x) = a*x^3 + b*x^2 + c*x + d
        //      given: q(u) = fu, q'(u) = gu
        //      given: q(v) = fv, q'(v) = gv
        // minimizer: solution of 3*a*x^2 + 2*b*x + c = 0
        // see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        const auto d1 = gu + gv - 3 * (fu - fv) / (u - v);
        const auto d2 = (v > u ? +1 : -1) * std::sqrt(d1 * d1 - gu * gv);
        return v - (v - u) * (gv + d2 - d1) / (gv - gu + 2 * d2);
}

static auto quadratic1(
        const scalar_t u, const scalar_t fu, const scalar_t gu,
        const scalar_t v, const scalar_t fv)
{
        // fit quadratic: q(x) = a*x^2 + b*x + c
        //      given: q(u) = fu, q'(u) = gu
        //      given: q(v) = fv
        // minimizer: -b/2a
        return u - gu * (u - v) * (u - v) / (2 * (gu * (u - v) - (fu - fv)));
}

static auto quadratic2(
        const scalar_t u, const scalar_t gu,
        const scalar_t v, const scalar_t gv)
{
        // fit quadratic: q(x) = a*x^2 + b*x + c
        //      given: q'(u) = gu
        //      given: q'(v) = gv
        // minimizer: -b/2a
        return (v * gu - u * gv) / (gu - gv);
}

static void update_interval(lsearch_step_t& l, lsearch_step_t& u, lsearch_step_t& t)
{
        // case a: l = l, u = t
        if (t.phi() > l.phi())
        {
                std::swap(u, t);
        }

        // case b: l = t, u = u
        else if (t.gphi() * (l.alpha() - t.alpha()) > 0)
        {
                std::swap(l, t);
        }

        // case c: l = t, u = l
        else
        {
                std::swap(l, t);
                std::swap(u, t);
        }
}

static auto trial_value_selection(
        const scalar_t l, const scalar_t fl, const scalar_t gl,
        const scalar_t u, const scalar_t fu, const scalar_t gu,
        const scalar_t t, const scalar_t ft, const scalar_t gt)
{
        // case 1
        if (ft > fl)
        {
                const auto mc = cubic(l, fl, gl, t, ft, gt);
                const auto mq = quadratic1(l, fl, gl, t, ft);

                return  (std::fabs(mc - l) < std::fabs(mq - l)) ?
                        (mc) :
                        ((mc + mq) / 2);
        }

        // case 2
        else if (gt * gl < 0)
        {
                const auto mc = cubic(l, fl, gl, t, ft, gt);
                const auto ms = quadratic2(l, gl, t, gt);

                return  (std::fabs(mc - t) >= std::fabs(ms - t)) ?
                        (mc) :
                        (ms);
        }

        // case 3
        else if (std::fabs(gt) <= std::fabs(gl))
        {
                const auto mc = cubic(l, fl, gl, t, ft, gt);
                const auto ms = quadratic2(l, gl, t, gt);

                const auto redefine = [&] (const auto mt)
                {
                        const auto gamma = scalar_t(0.66);
                        return  (t > l) ?
                                std::min(mt, t + gamma * (u - t)) :
                                std::max(mt, t + gamma * (u - t));
                };

                return  (std::fabs(mc - t) < std::fabs(ms - t)) ?
                        redefine(mc) :
                        redefine(ms);
        }

        // case 4
        else
        {
                return cubic(u, fu, gu, t, ft, gt);
        }
}

lsearch_morethuente_t::lsearch_morethuente_t(const scalar_t c1, const scalar_t c2) :
        m_c1(c1),
        m_c2(c2)
{
}

lsearch_step_t lsearch_morethuente_t::get(const lsearch_step_t& step0, const scalar_t t0)
{
        // NB: the implementation follows the notation from the original paper of More&Thuente (1994)
        scalar_t l = 0, t = t0, u = std::numeric_limits<scalar_t>::infinity();
        lsearch_step_t stept = step0;

        for (auto i = 0; i < m_max_iterations && t > lsearch_step_t::minimum() && t < lsearch_step_t::maximum(); ++ i)
        {
                // check convergence (sufficient decrease & curvature)
                if (!stept.update(t))
                {
                        return step0;
                }
                if (stept.has_armijo(m_c1) && stept.has_strong_wolfe(m_c2))
                {
                        return stept;
                }

                // todo
                break;
        }

        // NOK, give up
        return step0;
}
