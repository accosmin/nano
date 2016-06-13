#include "state.h"

namespace nano
{
        state_t::state_t(const tensor_size_t size) :
                x(size), g(size), d(size),
                f(std::numeric_limits<scalar_t>::max()),
                m_iterations(0),
                m_fcalls(0),
                m_gcalls(0),
                m_status(status::max_iters)
        {
        }

        bool state_t::update(const state_t& state)
        {
                const bool better = state < (*this);
                if (better)
                {
                        x = state.x;
                        g = state.g;
                        d = state.d;
                        f = state.f;
                }

                m_iterations = state.m_iterations;
                m_fcalls = state.m_fcalls;
                m_gcalls = state.m_gcalls;

                return better;
        }
}
