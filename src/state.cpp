#include "state.h"
#include "problem.h"

namespace nano
{
        state_t::state_t(const tensor_size_t size) :
                x(size), g(size), d(size),
                f(std::numeric_limits<scalar_t>::max()),
                m_iterations(0),
                m_fcalls(0),
                m_gcalls(0),
                m_status(opt_status::max_iters)
        {
        }

        state_t::state_t(const problem_t& problem, const vector_t& x0) :
                state_t(problem.size())
        {
                x = x0;
                f = problem.vgrad(x, g);
        }

        void state_t::update(const problem_t& problem, const vector_t& xx)
        {
                x = xx;
                f = problem.vgrad(x, g);

                m_iterations ++;
                m_fcalls = problem.fcalls();
                m_gcalls = problem.gcalls();
        }

        void state_t::update(const problem_t& problem, const scalar_t t)
        {
                x.noalias() += t * d;
                f = problem.vgrad(x, g);

                m_iterations ++;
                m_fcalls = problem.fcalls();
                m_gcalls = problem.gcalls();
        }

        void state_t::update(const problem_t& problem, const scalar_t t, const scalar_t ft, const vector_t& gt)
        {
                x.noalias() += t * d;
                f = ft;
                g = gt;

                m_iterations ++;
                m_fcalls = problem.fcalls();
                m_gcalls = problem.gcalls();
        }
}
