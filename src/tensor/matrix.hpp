#ifndef NANOCV_TENSOR_MATRIX_HPP
#define NANOCV_TENSOR_MATRIX_HPP

#include <eigen3/Eigen/Core>
#include <boost/serialization/vector.hpp>
#include <type_traits>

namespace ncv
{
        namespace tensor
        {
                ///
                /// vector
                ///
                template
                <
                        typename tvalue_,
                        typename tvalue = typename std::remove_const<tvalue_>::type
                >
                struct vector_types_t
                {
                        typedef tvalue                                  tscalar;

                        typedef Eigen::Matrix
                        <
                                tvalue,
                                Eigen::Dynamic,
                                1,
                                Eigen::ColMajor
                        >                                               tvector;
                        typedef std::vector<tvector>                    tvectors;
                        typedef typename tvectors::const_iterator       tvectors_const_it;
                        typedef typename tvectors::iterator             tvectors_it;
                };

                ///
                /// fixed size vector
                ///
                template
                <
                        typename tvalue_,
                        std::size_t trows,
                        typename tvalue = typename std::remove_const<tvalue_>::type
                >
                struct fixed_size_vector_types_t
                {
                        typedef tvalue                                  tscalar;

                        typedef Eigen::Matrix
                        <
                                tvalue,
                                trows,
                                1,
                                Eigen::ColMajor
                        >                                               tvector;
                        typedef std::vector<tvector>                    tvectors;
                        typedef typename tvectors::const_iterator       tvectors_const_it;
                        typedef typename tvectors::iterator             tvectors_it;
                };

                ///
                /// map data to vectors
                ///
                template
                <
                        typename tvalue_,
                        typename tsize,
                        typename tvalue = typename std::remove_const<tvalue_>::type,
                        typename tresult = Eigen::Map<typename vector_types_t<tvalue>::tvector>
                >
                tresult make_vector(tvalue_* data, tsize rows)
                {
                        return tresult((tvalue*)data, rows);
                }

                ///
                /// matrix
                ///
                template
                <
                        typename tvalue_,
                        typename tvalue = typename std::remove_const<tvalue_>::type
                >
                struct matrix_types_t
                {
                        typedef tvalue                                  tscalar;

                        typedef Eigen::Matrix
                        <       tvalue,
                                Eigen::Dynamic,
                                Eigen::Dynamic,
                                Eigen::RowMajor
                        >                                               tmatrix;
                        typedef std::vector<tmatrix>                    tmatrices;
                        typedef typename tmatrices::const_iterator      tmatrices_const_it;
                        typedef typename tmatrices::iterator            tmatrices_it;
                        typedef typename tmatrix::Index                 tindex;
                };

                ///
                /// fixed size matrix
                ///
                template
                <
                        typename tvalue_,
                        std::size_t trows,
                        std::size_t tcols,
                        typename tvalue = typename std::remove_const<tvalue_>::type
                >
                struct fixed_size_matrix_types_t
                {
                        typedef tvalue                                  tscalar;

                        typedef Eigen::Matrix
                        <       tvalue,
                                trows,
                                tcols,
                                Eigen::RowMajor
                        >                                               tmatrix;
                        typedef std::vector<tmatrix>                    tmatrices;
                        typedef typename tmatrices::const_iterator      tmatrices_const_it;
                        typedef typename tmatrices::iterator            tmatrices_it;
                        typedef typename tmatrix::Index                 tindex;
                };

                ///
                /// map data to matrices
                ///
                template
                <
                        typename tvalue_,
                        typename tsize,
                        typename tvalue = typename std::remove_const<tvalue_>::type,
                        typename tresult = Eigen::Map<typename matrix_types_t<tvalue>::tmatrix>
                >
                tresult make_matrix(tvalue_* data, tsize rows, tsize cols)
                {
                        return tresult((tvalue*)data, rows, cols);
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

#endif // NANOCV_TENSOR_MATRIX_HPP

