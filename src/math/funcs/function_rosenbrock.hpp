#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Rosenbrock test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_rosenbrock_t : public function_t
        {
                explicit function_rosenbrock_t(const tsize dims) :
                        m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "Rosenbrock" + std::to_string(m_dims) + "D";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                scalar_t fx = 0;
                                for (tsize i = 0; i + 1 < m_dims; i ++)
                                {
                                        fx += 100 * nano::square(x(i + 1) - x(i) * x(i)) + nano::square(x(i) - 1);
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
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

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return util::norm(x) < scalar_t(2.4);
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        {
                                const vector_t xmin = vector_t::Ones(m_dims);

                                if (util::distance(x, xmin) < epsilon)
                                {
                                        return true;
                                }
                        }

                        if (m_dims >= 4 && m_dims <= 7)
                        {
                                vector_t xmin = vector_t::Ones(m_dims);
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
