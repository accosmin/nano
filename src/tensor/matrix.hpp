#ifndef NANOCV_TENSOR_MATRIX_HPP
#define NANOCV_TENSOR_MATRIX_HPP

#include <eigen3/Eigen/Core>
#include <boost/serialization/vector.hpp>

namespace ncv
{
        namespace tensor
        {
                ///
                /// vector
                ///
                template
                <
                        typename tvalue
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

                        typedef Eigen::Map<tvector>                     tmap;
                };

                ///
                /// fixed size vector
                ///
                template
                <
                        typename tvalue,
                        std::size_t trows
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

                        typedef Eigen::Map<tvector>                     tmap;
                };

                ///
                /// matrix
                ///
                template
                <
                        typename tvalue
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

                        typedef Eigen::Map<tmatrix>                     tmap;
                };

                ///
                /// fixed size matrix
                ///
                template
                <
                        typename tvalue,
                        std::size_t trows,
                        std::size_t tcols
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

                        typedef Eigen::Map<tmatrix>                     tmap;
                };
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

