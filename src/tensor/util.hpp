#ifndef NANOCV_TENSOR_UTIL_HPP
#define NANOCV_TENSOR_UTIL_HPP

#include <eigen3/Eigen/Core>
#include <boost/serialization/vector.hpp>
#include <type_traits>

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
                        std::copy(t.data(), t.data() + t.size(), data);
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
                        std::copy(data, data + t.size(), t.data());
                        return data + t.size();
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

#endif // NANOCV_TENSOR_UTIL_HPP

