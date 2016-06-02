#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Cauchy test functions
        ///
        struct function_cauchy_t : public function_t
        {
                explicit function_cauchy_t(const tsize dims) :
                        m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "Cauchy" + std::to_string(m_dims) + "D";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                return std::log((1.0 + x.array().square()).prod());
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx = (2 * x.array()) / (1 + x.array().square());

                                return fn_fval(x);
                        };

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t&) const override
                {
                        return true;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
        };
}
