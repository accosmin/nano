#include "problem.h"
#include "math/epsilon.h"

namespace nano
{
        problem_t::problem_t(
                const opsize_t& opsize,
                const opfval_t& opfval,
                const opgrad_t& opgrad) :
                problem_t(opsize, opfval, opgrad, opfval, opgrad, 1)
        {
        }

        problem_t::problem_t(
                const opsize_t& opsize,
                const opfval_t& opfval) :
                problem_t(opsize, opfval, opgrad_t(), opfval, opgrad_t(), 1)
        {
        }

        problem_t::problem_t(
                const opsize_t& opsize,
                const opfval_t& opfval,
                const opgrad_t& opgrad,
                const opfval_t& stoch_opfval,
                const opgrad_t& stoch_opgrad,
                const size_t stoch_ratio) :
                m_opsize(opsize),
                m_opfval(opfval),
                m_opgrad(opgrad),
                m_stoch_ratio(std::max(size_t(1), stoch_ratio)),
                m_stoch_opfval(stoch_opfval),
                m_stoch_opgrad(stoch_opgrad),
                m_fcalls(0), m_stoch_fcalls(0),
                m_gcalls(0), m_stoch_gcalls(0)
        {
        }

        void problem_t::clear() const
        {
                m_fcalls = m_stoch_fcalls = 0;
                m_gcalls = m_stoch_gcalls = 0;
        }

        tensor_size_t problem_t::size() const
        {
                assert(m_opsize);

                return m_opsize();
        }

        scalar_t problem_t::value(const vector_t& x) const
        {
                assert(m_opfval);

                m_fcalls ++;
                return m_opfval(x);
        }

        scalar_t problem_t::stoch_value(const vector_t& x) const
        {
                assert(m_stoch_opfval);
                assert(x.size() == size());

                m_stoch_fcalls ++;
                return m_stoch_opfval(x);
        }

        scalar_t problem_t::vgrad(const vector_t& x, vector_t& g) const
        {
                assert(x.size() == size());

                if (m_opgrad)
                {
                        m_fcalls ++;
                        m_gcalls ++;
                        return m_opgrad(x, g);
                }
                else
                {
                        eval_grad(x, g);
                        return value(x);
                }
        }

        scalar_t problem_t::stoch_vgrad(const vector_t& x, vector_t& g) const
        {
                assert(m_stoch_opgrad);
                assert(x.size() == size());

                m_stoch_fcalls ++;
                m_stoch_gcalls ++;
                return m_stoch_opgrad(x, g);
        }

        scalar_t problem_t::grad_accuracy(const vector_t& x) const
        {
                assert(m_stoch_opgrad);
                assert(x.size() == size());

                vector_t gx;
                const scalar_t fx = m_opgrad(x, gx);

                vector_t gx_approx;
                eval_grad(x, gx_approx);

                return  (gx - gx_approx).lpNorm<Eigen::Infinity>() /
                        (scalar_t(1) + std::fabs(fx));
        }

        void problem_t::eval_grad(const vector_t& x, vector_t& g) const
        {
                // accuracy epsilon as defined in:
                //      see "Numerical optimization", Nocedal & Wright, 2nd edition, p.197
                const auto dx =
                        std::sqrt(std::numeric_limits<scalar_t>::epsilon()) *
                        (1 + x.lpNorm<Eigen::Infinity>());

                const auto n = size();

                vector_t xp = x, xn = x;

                g.resize(n);
                for (vector_t::Index i = 0; i < n; i ++)
                {
                        if (i > 0)
                        {
                                xp(i - 1) -= dx;
                                xn(i - 1) += dx;
                        }
                        xp(i) += dx;
                        xn(i) -= dx;

                        const auto dfi = m_opfval(xp) - m_opfval(xn);
                        const auto dxi = xp(i) - xn(i);
                        g(i) = static_cast<scalar_t>(dfi / dxi);
                }
        }

        bool problem_t::is_convex(const vector_t& x1, const vector_t& x2, const int steps) const
        {
                assert(steps > 2);

                const auto f1 = value(x1);
                assert(std::isfinite(f1));

                const auto f2 = value(x2);
                assert(std::isfinite(f2));

                for (int step = 1; step < steps; step ++)
                {
                        const auto t1 = scalar_t(step) / scalar_t(steps);
                        const auto t2 = scalar_t(1) - t1;

                        const auto ft = value(t1 * x1 + t2 * x2);
                        assert(std::isfinite(ft));

                        if (ft > (t1 * f1 + t2 * f2) + epsilon0<scalar_t>())
                        {
                                return false;
                        }
                }

                return true;
        }

        size_t problem_t::fcalls() const
        {
                return m_fcalls + m_stoch_fcalls / m_stoch_ratio;
        }

        size_t problem_t::gcalls() const
        {
                return m_gcalls + m_stoch_gcalls / m_stoch_ratio;
        }
}

