#pragma once

#include "function.hpp"
#include "math/numeric.hpp"

namespace func
{
        ///
        /// \brief create Rosenbrock test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        template
        <
                typename tscalar
        >
        struct function_rosenbrock_t : public function_t<tscalar>
        {
                typedef typename function_t<tscalar>::tsize     tsize;
                typedef typename function_t<tscalar>::tvector   tvector;
                typedef typename function_t<tscalar>::tproblem  tproblem;                
                
                explicit function_rosenbrock_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "Rosenbrock" + std::to_string(m_dims) + "D";
                }

                virtual tproblem problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                tscalar fx = 0;
                                for (tsize i = 0; i + 1 < m_dims; i ++)
                                {
                                        fx += 100.0 * math::square(x(i + 1) - x(i) * x(i)) + math::square(x(i) - 1);
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx.resize(m_dims);
                                gx.setZero();
                                for (tsize i = 0; i + 1 < m_dims; i ++)
                                {
                                        gx(i) += 2.0 * (x(i) - 1);
                                        gx(i) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i)) * (- 2.0 * x(i));
                                        gx(i + 1) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i));
                                }

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return norm(x) < 2.4;
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {

                        {
                                const tvector xmin = tvector::Ones(m_dims);

                                if (distance(x, xmin) < epsilon)
                                {
                                        return true;
                                }
                        }

                        if (m_dims >= 4 && m_dims <= 7)
                        {
                                tvector xmin = tvector::Ones(m_dims);
                                xmin(0) = -1;

                                if (distance(x, xmin) < epsilon)
                                {
                                        return true;
                                }
                        }

                        return false;
                }

                tsize   m_dims;
        };
}
