#ifndef NANOCV_TENSOR_MATRIX_HPP
#define NANOCV_TENSOR_MATRIX_HPP

#include <eigen3/Eigen/Core>
#include <boost/serialization/vector.hpp>

namespace ncv
{
        namespace tensor
        {
                //-------------------------------------------------------------------------------------------------
                // vector
                //-------------------------------------------------------------------------------------------------

                template
                <
                        typename tvalue
                >
                struct vector_types_t
                {
                        typedef tvalue                                  type_t;

                        typedef Eigen::Matrix
                        <
                                tvalue,
                                Eigen::Dynamic,
                                1,
                                Eigen::ColMajor
                        >                                               vector_t;
                        typedef std::vector<vector_t>                   vectors_t;
                        typedef typename vectors_t::const_iterator      vectors_const_it;
                        typedef typename vectors_t::iterator            vectors_it;
                };

                //-------------------------------------------------------------------------------------------------
                // fixed size vector
                //-------------------------------------------------------------------------------------------------

                template
                <
                        typename tvalue,
                        std::size_t trows
                >
                struct fixed_size_vector_types_t
                {
                        typedef tvalue                                  type_t;

                        typedef Eigen::Matrix
                        <
                                tvalue,
                                trows,
                                1,
                                Eigen::ColMajor
                        >                                               vector_t;
                        typedef std::vector<vector_t>                   vectors_t;
                        typedef typename vectors_t::const_iterator      vectors_const_it;
                        typedef typename vectors_t::iterator            vectors_it;
                };

                //-------------------------------------------------------------------------------------------------
                // matrix
                //-------------------------------------------------------------------------------------------------

                template
                <
                        typename tvalue
                >
                struct matrix_types_t
                {
                        typedef tvalue                                  type_t;

                        typedef Eigen::Matrix
                        <       tvalue,
                                Eigen::Dynamic,
                                Eigen::Dynamic,
                                Eigen::RowMajor
                        >                                               matrix_t;
                        typedef std::vector<matrix_t>                   matrices_t;
                        typedef typename matrices_t::const_iterator     matrices_const_it;
                        typedef typename matrices_t::iterator           matrices_it;
                        typedef typename matrix_t::Index                index_t;
                };

                //-------------------------------------------------------------------------------------------------
                // fixed size matrix
                //-------------------------------------------------------------------------------------------------

                template
                <
                        typename tvalue,
                        std::size_t trows,
                        std::size_t tcols
                >
                struct fixed_size_matrix_types_t
                {
                        typedef tvalue                                  type_t;

                        typedef Eigen::Matrix
                        <       tvalue,
                                trows,
                                tcols,
                                Eigen::RowMajor
                        >                                               matrix_t;
                        typedef std::vector<matrix_t>                   matrices_t;
                        typedef typename matrices_t::const_iterator     matrices_const_it;
                        typedef typename matrices_t::iterator           matrices_it;
                        typedef typename matrix_t::Index                index_t;
                };
        }
}

namespace boost
{
        namespace serialization
        {
                //-------------------------------------------------------------------------------------------------
                // serialize matrices and vectors
                //-------------------------------------------------------------------------------------------------

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

                //-------------------------------------------------------------------------------------------------
        }
}

#endif // NANOCV_TENSOR_MATRIX_HPP

