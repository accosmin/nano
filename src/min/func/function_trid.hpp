#pragma once

#include "util.hpp"

namespace func
{
        ///
        /// \brief create Trid test functions
        ///
        template
        <
                typename tscalar_
        >
        struct function_trid_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                explicit function_trid_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                std::string name() const
                {
                        return "Trid" + std::to_string(m_dims) + "D";
                }

                tproblem problem() const
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

                bool is_valid(const tvector& x) const
                {
                        return util::norm(x) < m_dims * m_dims;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
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
