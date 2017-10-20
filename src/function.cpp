#include "function.h"
#include "math/epsilon.h"

using namespace nano;

function_t::function_t(const char* name,
        const tensor_size_t size, const tensor_size_t min_size, const tensor_size_t max_size,
        const convexity convex,
        const scalar_t domain) :
        m_name(name),
        m_size(size), m_min_size(min_size), m_max_size(max_size),
        m_convex(convex),
        m_domain(domain),
        m_fcalls(0), m_stoch_fcalls(0),
        m_gcalls(0), m_stoch_gcalls(0)
{
}

scalar_t function_t::eval(const vector_t& x, vector_t* gx) const
{
        assert(x.size() == size());

        m_fcalls ++;
        if (gx)
        {
                m_gcalls ++;
                gx->resize(size());
        }

        const auto f = vgrad(x, gx);
        assert(!gx || gx->size() == size());
        return f;
}

scalar_t function_t::stoch_eval(const vector_t& x, vector_t* gx) const
{
        assert(x.size() == size());

        m_stoch_fcalls ++;
        if (gx)
        {
                m_stoch_gcalls ++;
                gx->resize(size());
        }

        const auto f = stoch_vgrad(x, gx);
        assert(!gx || gx->size() == size());
        return f;
}

scalar_t function_t::grad_accuracy(const vector_t& x) const
{
        assert(x.size() == size());

        const auto n = size();

        vector_t gx(n);
        vector_t gx_approx(n);
        vector_t xp = x, xn = x;

        // analytical gradient
        const auto fx = vgrad(x, &gx);
        assert(gx.size() == size());

        // finite-difference approximated gradient
        //      see "Numerical optimization", Nocedal & Wright, 2nd edition, p.197
        const auto finite_difference = [&] (const scalar_t dx)
        {
                for (auto i = 0; i < n; i ++)
                {
                        xp = x; xp(i) += dx;
                        xn = x; xn(i) -= dx;

                        const auto dfi = vgrad(xp, nullptr) - vgrad(xn, nullptr);
                        const auto dxi = xp(i) - xn(i);
                        gx_approx(i) = dfi / dxi;
                }

                return (gx - gx_approx).lpNorm<Eigen::Infinity>() / (1 + std::fabs(fx));
        };

        return finite_difference(epsilon2<scalar_t>());
}

bool function_t::is_convex(const vector_t& x1, const vector_t& x2, const int steps) const
{
        assert(steps > 2);
        assert(x1.size() == size());
        assert(x2.size() == size());

        const auto f1 = vgrad(x1, nullptr);
        assert(std::isfinite(f1));

        const auto f2 = vgrad(x2, nullptr);
        assert(std::isfinite(f2));

        for (int step = 1; step < steps; step ++)
        {
                const auto t1 = scalar_t(step) / scalar_t(steps);
                const auto t2 = scalar_t(1) - t1;
                const auto ftc = t1 * f1 + t2 * f2;

                const auto ft = vgrad(t1 * x1 + t2 * x2, nullptr);
                if (std::isfinite(ft) && ft > ftc + epsilon0<scalar_t>() * (1 + std::fabs(ftc)))
                {
                        return false;
                }
        }

        return true;
}

size_t function_t::fcalls() const
{
        return m_fcalls + m_stoch_fcalls / stoch_ratio();
}

size_t function_t::gcalls() const
{
        return m_gcalls + m_stoch_gcalls / stoch_ratio();
}

string_t function_t::name() const
{
        return string_t(m_name) + "[" + std::to_string(size()) + "D]";
}

bool function_t::is_valid(const vector_t& x) const
{
        return x.lpNorm<Eigen::Infinity>() < m_domain;
}
