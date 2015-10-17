#pragma once

#include "util.hpp"

namespace func
{
        ///
        /// \brief create Zakharov test functions
        ///
        template
        <
                typename tscalar_
        >
        struct function_zakharov_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                explicit function_zakharov_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                std::string name() const
                {
                        return "Zakharov" + std::to_string(m_dims) + "D";
                }

                tproblem problem() const
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                tscalar u = 0, v = 0;
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        u += util::square(x(i));
                                        v += 0.5 * i * x(i);
                                }

                                return u + util::square(v) + util::quartic(v);
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                tscalar u = 0, v = 0;
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        u += util::square(x(i));
                                        v += 0.5 * i * x(i);
                                }

                                gx.resize(m_dims);
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        gx(i) = 2 * x(i) + (2 * v + 4 * util::cube(v)) * 0.5 * i;
                                }

                                return u + util::square(v) + util::quartic(v);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                bool is_valid(const tvector& x) const
                {
                        return -5.0 < x.minCoeff() && x.maxCoeff() < 10.0;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
                {
                        return util::distance(x, tvector::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
        };  
}
