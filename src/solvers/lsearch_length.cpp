#include <utility>
#include "math/cubic.h"
#include "math/quadratic.h"
#include "lsearch_length.h"

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
