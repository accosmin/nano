#include "ls_step.h"

namespace nano
{
        ls_step_t::ls_step_t(const function_t& function, const state_t& state) :
                m_function(function),
                m_state(state),
                m_gphi0(state.d.dot(state.g)),
                m_alpha(0),
                m_func(state.f),
                m_grad(state.g),
                m_gphi(m_gphi0)
        {
        }

        scalar_t ls_step_t::minimum()
        {
                return scalar_t(10) * std::numeric_limits<scalar_t>::epsilon();
        }

        scalar_t ls_step_t::maximum()
        {
                return scalar_t(1) / minimum();
        }

        bool ls_step_t::update(const scalar_t alpha)
        {
                if (!std::isfinite(alpha))
                {
                        return false;
                }
                else
                {
                        m_alpha = alpha;
                        m_func = m_function.get().eval(m_state.get().x + m_alpha * m_state.get().d, &m_grad);
                        m_gphi = m_grad.dot(m_state.get().d);
                        return operator bool();
                }
        }

        bool ls_step_t::has_armijo(const scalar_t c1) const
        {
                return phi() <= phi0() + alpha() * c1 * gphi0();
        }

        bool ls_step_t::has_wolfe(const scalar_t c2) const
        {
                return gphi() >= +c2 * gphi0();
        }

        bool ls_step_t::has_strong_wolfe(const scalar_t c2) const
        {
                return  gphi() >= +c2 * gphi0() &&
                        gphi() <= -c2 * gphi0();
        }

        bool ls_step_t::has_approx_wolfe(const scalar_t c1, const scalar_t c2, const scalar_t epsilon) const
        {
                return  (2 * c1 - 1) * gphi0() >= gphi() &&
                        gphi() >= +c2 * gphi0() &&
                        phi() <= approx_phi(epsilon);
        }

        scalar_t ls_step_t::alpha() const
        {
                return m_alpha;
        }

        scalar_t ls_step_t::phi() const
        {
                return m_func;
        }

        scalar_t ls_step_t::approx_phi(const scalar_t epsilon) const
        {
                return phi0() + epsilon;
        }

        scalar_t ls_step_t::phi0() const
        {
                return m_state.get().f;
        }

        scalar_t ls_step_t::gphi() const
        {
                return m_gphi;
        }

        scalar_t ls_step_t::gphi0() const
        {
                return m_gphi0;
        }

        scalar_t ls_step_t::func() const
        {
                return phi();
        }

        const vector_t& ls_step_t::grad() const
        {
                return m_grad;
        }

        ls_step_t::operator bool() const
        {
                return  std::isfinite(alpha()) &&
                        std::isfinite(phi()) &&
                        std::isfinite(gphi());
        }
}
