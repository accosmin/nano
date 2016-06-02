#pragma once

#include "function.hpp"
#include <cmath>

namespace nano
{
        ///
        /// \brief create Bohachevsky test functions
        ///
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
                explicit function_bohachevsky_t(const btype type) :
                        m_type(type)
                {
                }

                virtual std::string name() const override
                {
                        switch (m_type)
                        {
                        case btype::one:        return "Bohachevsky1";
                        case btype::two:        return "Bohachevsky2";
                        case btype::three:      return "Bohachevsky3";
                        default:                return "Bohachevskyx";
                        }
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return vector_t::Index(2);
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);

                                const auto pi = std::atan2(scalar_t(0.0), scalar_t(-0.0));
                                const auto p1 = 3 * pi * x1;
                                const auto p2 = 4 * pi * x2;

                                const auto u = x1 * x1 + 2 * x2 * x2;

                                scalar_t fx = 0;
                                switch (m_type)
                                {
                                case btype::one:
                                        fx = u - scalar_t(0.3) * std::cos(p1) - scalar_t(0.4) * std::cos(p2) + scalar_t(0.7);
                                        break;

                                case btype::two:
                                        fx = u - scalar_t(0.3) * std::cos(p1) * std::cos(p2) + scalar_t(0.3);
                                        break;

                                case btype::three:
                                        fx = u - scalar_t(0.3) * std::cos(p1 + p2) + scalar_t(0.3);
                                        break;

                                default:
                                        break;
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const auto x1 = x(0);
                                const auto x2 = x(1);

                                const auto pi = std::atan2(scalar_t(0.0), scalar_t(-0.0));
                                const auto p1 = 3 * pi * x1;
                                const auto p2 = 4 * pi * x2;

                                gx.resize(2);
                                switch (m_type)
                                {
                                case btype::one:
                                        gx(0) = 2 * x1 + scalar_t(0.9) * std::sin(p1) * pi;
                                        gx(1) = 4 * x2 + scalar_t(1.6) * std::sin(p2) * pi;
                                        break;

                                case btype::two:
                                        gx(0) = 2 * x1 + scalar_t(0.9) * std::sin(p1) * pi * std::cos(p2);
                                        gx(1) = 4 * x2 + scalar_t(1.2) * std::sin(p2) * pi * std::cos(p1);
                                        break;

                                case btype::three:
                                        gx(0) = 2 * x1 + scalar_t(0.9) * std::sin(p1 + p2) * pi;
                                        gx(1) = 4 * x2 + scalar_t(1.2) * std::sin(p1 + p2) * pi;
                                        break;

                                default:
                                        break;
                                }

                                return fn_fval(x);
                        };

                        return tproblem{fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return scalar_t(-100) < x.minCoeff() && x.maxCoeff() < scalar_t(100);
                }

                virtual bool is_minima(const vector_t&, const scalar_t) const override
                {
                        // NB: there are quite a few local minima that are not easy to compute!
                        return true;
//                        return distance(x, vector_t::Zero(2)) < epsilon;
                }

                btype   m_type;
        };
}
