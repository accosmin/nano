#pragma once

#include "util.hpp"
#include "function.hpp"

namespace math
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
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

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
                                        fx += 100 * math::square(x(i + 1) - x(i) * x(i)) + math::square(x(i) - 1);
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx.resize(m_dims);
                                gx.setZero();
                                for (tsize i = 0; i + 1 < m_dims; i ++)
                                {
                                        gx(i) += 2 * (x(i) - 1);
                                        gx(i) += 100 * 2 * (x(i + 1) - x(i) * x(i)) * (- 2 * x(i));
                                        gx(i + 1) += 100 * 2 * (x(i + 1) - x(i) * x(i));
                                }

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return util::norm(x) < tscalar(2.4);
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        {
                                const tvector xmin = tvector::Ones(m_dims);

                                if (util::distance(x, xmin) < epsilon)
                                {
                                        return true;
                                }
                        }

                        if (m_dims >= 4 && m_dims <= 7)
                        {
                                tvector xmin = tvector::Ones(m_dims);
                                xmin(0) = -1;

                                if (util::distance(x, xmin) < epsilon)
                                {
                                        return true;
                                }
                        }

                        return false;
                }

                tsize   m_dims;
        };
}
