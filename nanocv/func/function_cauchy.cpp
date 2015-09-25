#include "function_cauchy.h"
#include "text/to_string.hpp"

namespace ncv
{
        struct function_cauchy_t : public function_t
        {
                explicit function_cauchy_t(const size_t dims)
                        :       m_dims(dims)
                {
                }

                virtual string_t name() const override
                {
                        return "Cauchy" + text::to_string(m_dims) + "D";
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const opt_opfval_t fn_fval = [=] (const vector_t& x)
                        {
                                return (1.0 + x.array().square()).log().sum();
                        };

                        const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx = (2.0 * x.array()) / (1.0 + x.array().square());

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const vector_t&) const override
                {
                        return true;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return distance(x, vector_t::Zero(m_dims)) < epsilon;
                }

                size_t  m_dims;
        };

        functions_t make_cauchy_funcs(size_t max_dims)
        {
                functions_t functions;

                for (size_t dims = 1; dims <= max_dims; dims *= 2)
                {
                        functions.push_back(std::make_shared<function_cauchy_t>(dims));
                }

                return functions;
        }
}
	
