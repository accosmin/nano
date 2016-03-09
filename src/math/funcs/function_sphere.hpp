#pragma once

#include "util.hpp"
#include "function.hpp"

namespace zob
{
        ///
        /// \brief create sphere test functions
        ///
        template 
        <
                typename tscalar
        >
        struct function_sphere_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                explicit function_sphere_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "sphere" + std::to_string(m_dims) + "D";
                }

                virtual tproblem problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                return x.array().square().sum();
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx = 2 * x;

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
