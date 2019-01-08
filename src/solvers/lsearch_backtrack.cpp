#include "lsearch_backtrack.h"

using namespace nano;

lsearch_step_t lsearch_backtrack_armijo_t::get(const lsearch_step_t& step0, const scalar_t t0)
{
        auto t = t0;
        auto step = step0;

        for (int i = 0; i < m_max_iterations && t > lsearch_step_t::minimum() && t < lsearch_step_t::maximum(); ++ i)
        {
                if (!step.update(t))
                {
                        return step0;
                }
                else if (!step.has_armijo(m_c1))
                {
                        t *= m_decrement;
                }
                else
                {
                        step.update(t);
                        return step;
                }
        }

        return step0;
}

lsearch_step_t lsearch_backtrack_wolfe_t::get(const lsearch_step_t& step0, const scalar_t t0)
{
        auto t = t0;
        auto step = step0;

        for (int i = 0; i < m_max_iterations && t > lsearch_step_t::minimum() && t < lsearch_step_t::maximum(); ++ i)
        {
                if (!step.update(t))
                {
                        return step0;
                }
                else if (!step.has_armijo(m_c1))
                {
                        t *= m_decrement;
                }
                else if (!step.has_wolfe(m_c2))
                {
                        t *= m_increment;
                }
                else
                {
                        step.update(t);
                        return step;
                }
        }

        return step0;
}

lsearch_step_t lsearch_backtrack_swolfe_t::get(const lsearch_step_t& step0, const scalar_t t0)
{
        auto t = t0;
        auto step = step0;

        for (int i = 0; i < m_max_iterations && t > lsearch_step_t::minimum() && t < lsearch_step_t::maximum(); ++ i)
        {
                if (!step.update(t))
                {
                        return step0;
                }
                else if (!step.has_armijo(m_c1))
                {
                        t *= m_decrement;
                }
                else if (!step.has_wolfe(m_c2))
                {
                        t *= m_increment;
                }
                else if (!step.has_strong_wolfe(m_c2))
                {
                        t *= m_decrement;
                }
                else
                {
                        step.update(t);
                        return step;
                }
        }

        return step0;
}
