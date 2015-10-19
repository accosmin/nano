#pragma once

#include <boost/serialization/vector.hpp>
#include "vector.hpp"

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
                tensor::map_vector(data, t.size()) = tensor::map_vector(t.data(), t.size());
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
                tensor::map_vector(t.data(), t.size()) = tensor::map_vector(data, t.size());
                return data + t.size();
        }
}

namespace boost
{
        namespace serialization
        {
                ///
                /// \brief serialize matrices and vectors
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
                        using tsize = typename Eigen::Matrix<tvalue, Rows, Cols, Options>::Index;

                        if (tarchive::is_saving::value)
                        {
                                tsize rows = mat.rows(), cols = mat.cols();
                                ar & rows; ar & cols;

                                for (tsize i = 0; i < mat.size(); i ++)
                                {
                                        ar & mat(i);
                                }
                        }

                        else
                        {
                                tsize rows = 0, cols = 0;
                                ar & rows; ar & cols;

                                mat.resize(rows, cols);
                                for (tsize i = 0; i < mat.size(); i ++)
                                {
                                        ar & mat(i);
                                }
                        }
                }                
        }
}

