#include "core/numeric.h"
#include "lsearch_nocedalwright.h"

using namespace nano;

bool lsearch_nocedalwright_t::zoom(const solver_state_t& state0, step_t& l, step_t& h, solver_state_t& state) const
{
        for (int i = 0; i < max_iterations() && std::fabs(l.t - h.t) > epsilon0<scalar_t>(); ++ i)
        {
                // interpolate the new trial
                if (!state.update(state0, interpolate(l, h)))
                {
                        return false;
                }

                // check sufficient decrease
                else if (!state.has_armijo(state0, c1()) || state.f >= l.f)
                {
                        h = state;
                }

                // check curvature
                else if (state.has_strong_wolfe(state0, c2()))
                {
                        return true;
                }

                if (state.dg() * (h.t - l.t) >= 0)
                {
                        h = l;
                }
                l = state;
        }

        return false;
}

bool lsearch_nocedalwright_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
        step_t stept = state0, stepp = state0;

        for (int i = 1; i < max_iterations() && t < stpmax(); ++ i)
        {
                // check sufficient decrease
                if (!state.update(state0, t))
                {
                        return false;
                }

                else if (!state.has_armijo(state0, c1()) || (stept.f >= stepp.f && i > 1))
                {
                        return zoom(state0, stepp, stept, state);
                }

                // check curvature
                else if (state.has_strong_wolfe(state0, c2()))
                {
                        return true;
                }

                if (stept.g >= scalar_t(0))
                {
                        return zoom(state0, stept, stepp, state);
                }

                stepp = stept;
                t *= 3;
        }

        return false;
}
