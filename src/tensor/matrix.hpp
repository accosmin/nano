#pragma once

#include <eigen3/Eigen/Core>
#include <type_traits>
#include <vector>

namespace ncv
{
        namespace tensor
        {
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
                        return tresult(data, rows, cols);
                }

                ///
                /// map data to matrices
                ///
                template
                <
                        typename tvalue_,
                        typename tsize,
                        typename tvalue = typename std::remove_const<tvalue_>::type,
                        typename tresult = Eigen::Map<const typename matrix_types_t<tvalue>::tmatrix>
                >
                tresult make_matrix(const tvalue_* data, tsize rows, tsize cols)
                {
                        return tresult(data, rows, cols);
                }
        }
}
