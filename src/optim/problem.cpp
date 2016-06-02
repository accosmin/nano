#include "problem.h"

namespace nano
{
        problem_t::problem_t(
                const opsize_t& opsize,
                const opfval_t& opfval,
                const opgrad_t& opgrad) :
                m_opsize(opsize),
                m_opfval(opfval),
                m_opgrad(opgrad),
                m_fcalls(0),
                m_gcalls(0)
        {
        }

        problem_t::problem_t(
                const opsize_t& opsize,
                const opfval_t& opfval) :
                problem_t(opsize, opfval, opgrad_t())
        {
        }

        void problem_t::clear() const
        {
                m_fcalls = 0;
                m_gcalls = 0;
        }

        vector_t::Index problem_t::size() const
        {
                return m_opsize();
        }

        scalar_t problem_t::operator()(const vector_t& x) const
        {
                m_fcalls ++;
                return m_opfval(x);
        }

        scalar_t problem_t::operator()(const vector_t& x, vector_t& g) const
        {
                if (m_opgrad)
                {
                        m_fcalls ++;
                        m_gcalls ++;
                        return m_opgrad(x, g);
                }
                else
                {
                        eval_grad(x, g);
                        return operator()(x);
                }
        }

        scalar_t problem_t::grad_accuracy(const vector_t& x) const
        {
                if (m_opgrad)
                {
                        vector_t gx;
                        const scalar_t fx = m_opgrad(x, gx);

                        vector_t gx_approx;
                        eval_grad(x, gx_approx);

                        return  (gx - gx_approx).template lpNorm<Eigen::Infinity>() /
                                (scalar_t(1) + std::fabs(fx));
                }
                else
                {
                        return scalar_t(0);
                }
        }

        void problem_t::eval_grad(const vector_t& x, vector_t& g) const
        {
                // accuracy epsilon as defined in:
                //      see "Numerical optimization", Nocedal & Wright, 2nd edition, p.197
                const auto dx = std::cbrt(scalar_t(10) * std::numeric_limits<scalar_t>::epsilon());

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

                        g(i) = (m_opfval(xp) - m_opfval(xn)) / (xp(i) - xn(i));
                }
        }
}

