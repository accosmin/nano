#include "function.h"
#include "math/epsilon.h"

namespace nano
{
        function_t::function_t():
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

                return vgrad(x, gx);
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

                return stoch_vgrad(x, gx);
        }

        scalar_t function_t::grad_accuracy(const vector_t& x) const
        {
                assert(x.size() == size());

                const auto n = size();

                // analytical gradient
                vector_t gx(n);
                const auto fx = vgrad(x, &gx);

                // finite-difference approximated gradient
                //      see "Numerical optimization", Nocedal & Wright, 2nd edition, p.197
                const auto dx =
                        std::sqrt(std::numeric_limits<scalar_t>::epsilon()) *
                        (1 + x.lpNorm<Eigen::Infinity>());

                vector_t gx_approx(n);
                vector_t xp = x, xn = x;
                for (auto i = 0; i < n; i ++)
                {
                        if (i > 0)
                        {
                                xp(i - 1) -= dx;
                                xn(i - 1) += dx;
                        }
                        xp(i) += dx;
                        xn(i) -= dx;

                        const auto dfi = vgrad(xp, nullptr) - vgrad(xn, nullptr);
                        const auto dxi = xp(i) - xn(i);
                        gx_approx(i) = static_cast<scalar_t>(dfi / dxi);
                }

                // return the relative difference between gradients
                return  (gx - gx_approx).lpNorm<Eigen::Infinity>() /
                        (scalar_t(1) + std::fabs(fx));
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

                        const auto ft = vgrad(t1 * x1 + t2 * x2, nullptr);
                        if (std::isfinite(ft) && ft > (1 + epsilon0<scalar_t>()) * (t1 * f1 + t2 * f2))
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
}

