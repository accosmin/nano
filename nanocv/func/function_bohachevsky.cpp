#include "function_bohachevsky.h"
#include "math/numeric.hpp"
#include <cmath>

namespace ncv
{
        namespace
        {
                enum btype
                {
                        one,
                        two,
                        three
                };
        }

        struct function_bohachevsky_t : public function_t
        {
                explicit function_bohachevsky_t(const btype type)
                        :       m_type(type)
                {
                }

                virtual string_t name() const override
                {
                        switch (m_type)
                        {
                        case btype::one:    return "Bohachevsky1";
                        case btype::two:    return "Bohachevsky2";
                        case btype::three:  return "Bohachevsky3";
                        default:                        return "Bohachevskyx";
                        }
                }

                virtual opt_problem_t problem() const override
                {
                        const opt_opsize_t fn_size = [=] ()
                        {
                                return 2;
                        };

                        const opt_opfval_t fn_fval = [=] (const opt_vector_t& x)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);

                                const opt_scalar_t pi = std::atan2(0.0, -0.0);
                                const auto p1 = 3 * pi * x1;
                                const auto p2 = 4 * pi * x2;

                                const auto u = x1 * x1 + 2 * x2 * x2;

                                opt_scalar_t fx = 0;
                                switch (m_type)
                                {
                                case btype::one:
                                        fx = u - 0.3 * std::cos(p1) - 0.4 * std::cos(p2) + 0.7;
                                        break;

                                case btype::two:
                                        fx = u - 0.3 * std::cos(p1) * std::cos(p2) + 0.3;
                                        break;

                                case btype::three:
                                        fx = u - 0.3 * std::cos(p1 + p2) + 0.3;
                                        break;

                                default:
                                        break;
                                }

                                return fx;
                        };

                        const opt_opgrad_t fn_grad = [=] (const opt_vector_t& x, opt_vector_t& gx)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);

                                const opt_scalar_t pi = std::atan2(0.0, -0.0);
                                const auto p1 = 3 * pi * x1;
                                const auto p2 = 4 * pi * x2;

                                gx.resize(2);
                                switch (m_type)
                                {
                                case btype::one:
                                        gx(0) = 2 * x1 + 0.9 * std::sin(p1) * pi;
                                        gx(1) = 4 * x2 + 1.6 * std::sin(p2) * pi;
                                        break;

                                case btype::two:
                                        gx(0) = 2 * x1 + 0.9 * std::sin(p1) * pi * cos(p2);
                                        gx(1) = 4 * x2 + 1.2 * std::sin(p2) * pi * cos(p1);
                                        break;

                                case btype::three:
                                        gx(0) = 2 * x1 + 0.9 * std::sin(p1 + p2) * pi;
                                        gx(1) = 4 * x2 + 1.2 * std::sin(p1 + p2) * pi;
                                        break;

                                default:
                                        break;
                                }

                                return fn_fval(x);
                        };

                        return opt_problem_t(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const opt_vector_t& x) const override
                {
                        return -100.0 < x.minCoeff() && x.maxCoeff() < 100.0;
                }

                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const override
                {
                        return distance(x, opt_vector_t::Zero(2)) < epsilon;
                }

                btype   m_type;
        };

        functions_t make_bohachevsky_funcs()
        {
                return
                {
                        std::make_shared<function_bohachevsky_t>(btype::one),
                        std::make_shared<function_bohachevsky_t>(btype::two),
                        std::make_shared<function_bohachevsky_t>(btype::three)
                };
        }
}
	
