#pragma once

#include <boost/serialization/vector.hpp>
#include <algorithm>
#include "vector.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief serialize a tensor
                ///
                template
                <
                        typename ttensor
                >
                typename ttensor::Scalar* save(const ttensor& t, typename ttensor::Scalar* data)
                {
                        tensor::make_vector(data, t.size()) = tensor::make_vector(t.data(), t.size());
                        return data + t.size();
                }

                ///
                /// \brief serialize a tensor
                ///
                template
                <
                        typename ttensor
                >
                const typename ttensor::Scalar* load(ttensor& t, const typename ttensor::Scalar* data)
                {
                        tensor::make_vector(t.data(), t.size()) = tensor::make_vector(data, t.size());
                        return data + t.size();
                }

                ///
                /// \brief transform coefficient-wise a matrix: out = op(in)
                ///
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

                ///
                /// \brief transform coefficient-wise a matrix: out = op(in1, in2)
                ///
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

                ///
                /// \brief transform coefficient-wise a matrix: out = op(in1, in2, in3)
                ///
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

namespace boost
{
        namespace serialization
        {
                ///
                /// serialize matrices and vectors
                ///
                template
                <
                        class tarchive,
                        class tvalue,
                        int Rows,
                        int Cols,
                        int Options
                >
                void serialize(tarchive& ar, Eigen::Matrix<tvalue, Rows, Cols, Options>& mat, const unsigned int)
                {
                        if (tarchive::is_saving::value)
                        {
                                int rows = mat.rows(), cols = mat.cols();
                                ar & rows; ar & cols;

                                for (int i = 0; i < mat.size(); i ++)
                                {
                                        ar & mat(i);
                                }
                        }

                        else
                        {
                                int rows = 0, cols = 0;
                                ar & rows; ar & cols;

                                mat.resize(rows, cols);
                                for (int i = 0; i < mat.size(); i ++)
                                {
                                        ar & mat(i);
                                }
                        }
                }                
        }
}

