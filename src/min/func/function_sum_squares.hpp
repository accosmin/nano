#pragma once

#include "function.hpp"

namespace func
{
        ///
        /// \brief create sum of squares test functions
        ///
        template 
        <
                typename tscalar
        >
        struct function_sum_squares_t : public function_t<tscalar>
        {
                typedef typename function_t<tscalar>::tsize     tsize;
                typedef typename function_t<tscalar>::tvector   tvector;
                typedef typename function_t<tscalar>::tproblem  tproblem;  
                
                explicit function_sum_squares_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "sum squares" + std::to_string(m_dims) + "D";
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
                                        fx += (i + 1) * x(i) * x(i);
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx.resize(m_dims);
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        gx(i) = 2.0 * (i + 1) * x(i);
                                }

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return util::norm(x) < 5.12;
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        return util::distance(x, tvector::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
        };
}
