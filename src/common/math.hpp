#ifndef NANOCV_CAST_H
#define NANOCV_CAST_H

#include <type_traits>
#include <algorithm>
#include <boost/algorithm/clamp.hpp>

namespace ncv
{
        namespace math
        {
                // forward boost functions
                using boost::algorithm::clamp;
                using boost::algorithm::clamp_range;

                // implementation detail
                namespace impl
                {
                        template
                        <
                                typename tround,
                                bool tround_integral,
                                typename tvalue,
                                bool tvalue_integral
                        >
                        struct cast
                        {
                                static tround dispatch(tvalue value)
                                {
                                        return static_cast<tround>(value);
                                }
                        };

                        template
                        <
                                typename tround,
                                typename tvalue
                        >
                        struct cast<tround, true, tvalue, false>
                        {
                                static tround dispatch(tvalue value)
                                {
                                        return static_cast<tround>(std::nearbyint(value));
                                }
                        };
                }

                // cast a value to another type (with rounding to the closest if necessary)
                template
                <
                        typename tround,
                        typename tvalue
                >
                tround cast(tvalue value)
                {
                        return  impl::cast<
                                tround, std::is_integral<tround>::value,
                                tvalue, std::is_integral<tvalue>::value>::dispatch(value);
                }

                // square a value
                template
                <
                        typename tvalue
                >
                tvalue square(tvalue value)
                {
                        return value * value;
                }

                template
                <
                        typename tvalue
                >
                tvalue cube(tvalue value)
                {
                        return value * square(value);
                }

                // transform coefficient-wise a matrix: out = op(in)
                template
                <
                        typename tin_matrix,
                        typename tout_matrix,
                        typename toperator
                >
                void transform(const tin_matrix& in, tout_matrix& out, toperator op)
                {
                        std::transform(in.data(), in.data() + in.size(), out.data(), op);
                }

                // transform coefficient-wise a matrix: out = op(in1, in2)
                template
                <
                        typename tin1_matrix,
                        typename tin2_matrix,
                        typename tout_matrix,
                        typename toperator
                >
                void transform(const tin1_matrix& in1, const tin2_matrix& in2, tout_matrix& out, toperator op)
                {
                        std::transform(in1.data(), in1.data() + in1.size(), in2.data(), out.data(), op);
                }

                // transform coefficient-wise a matrix: out = op(in1, in2, in3)
                template
                <
                        typename tin1_matrix,
                        typename tin2_matrix,
                        typename tin3_matrix,
                        typename tout_matrix,
                        typename toperator
                >
                void transform(const tin1_matrix& in1, const tin2_matrix& in2, const tin3_matrix& in3,
                        tout_matrix& out, toperator op)
                {
                        auto in1_it = in1.data(), in1_end = in1.data() + in1.size();
                        auto in2_it = in2.data();
                        auto in3_it = in3.data();
                        auto out_it = out.data();

                        for ( ; in1_it != in1_end; ++ in1_it, ++ in2_it, ++ in3_it, ++ out_it)
                        {
                                *out_it = op(*in1_it, *in2_it, *in3_it);
                        }
                }
        }
}

#endif // NANOCV_CAST_H

