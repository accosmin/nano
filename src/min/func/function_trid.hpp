#pragma once

#include "util.hpp"
#include "function.hpp"

namespace min
{
        ///
        /// \brief create Trid test functions
        ///
        template
        <
                typename tscalar
        >
        struct function_trid_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;
                
                explicit function_trid_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "Trid" + std::to_string(m_dims) + "D";
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
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        fx += util::square(x(i) - 1.0);
                                }
                                for (tsize i = 1; i < m_dims; i ++)
                                {
                                        fx -= x(i) * x(i - 1);
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx.resize(m_dims);
                                gx.setZero();
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        gx(i) += 2.0 * (x(i) - 1.0);
                                }
                                for (tsize i = 1; i < m_dims; i ++)
                                {
                                        gx(i) -= x(i - 1);
                                        gx(i - 1) -= x(i);
                                }

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return util::norm(x) < tscalar(1) + m_dims * m_dims;
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        tvector xmin(m_dims);
                        for (tsize d = 0; d < m_dims; d ++)
                        {
                                xmin(d) = (d + 1.0) * (m_dims - d);
                        }

                        return util::distance(x, xmin) < epsilon;
                }

                tsize   m_dims;
        };
}
