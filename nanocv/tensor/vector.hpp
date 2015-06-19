#pragma once

#include <eigen3/Eigen/Core>
#include <type_traits>
#include <vector>

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief vector types
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
                        typedef typename tvector::Index                 tindex;
                };

                ///
                /// \brief fixed size vector types
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
                        typedef typename tvector::Index                 tindex;
                };

                ///
                /// \brief map non-constant data to vectors
                ///
                template
                <
                        typename tvalue_,
                        typename tsize,
                        typename tvalue = typename std::remove_const<tvalue_>::type,
                        typename tresult = Eigen::Map<typename vector_types_t<tvalue>::tvector>
                >
                tresult map_vector(tvalue_* data, tsize rows)
                {
                        return tresult(data, rows);
                }
                
                ///
                /// \brief map constant data to vectors
                ///
                template
                <
                        typename tvalue_,
                        typename tsize,
                        typename tvalue = typename std::remove_const<tvalue_>::type,
                        typename tresult = Eigen::Map<const typename vector_types_t<tvalue>::tvector>
                >
                tresult map_vector(const tvalue_* data, tsize rows)
                {
                        return tresult(data, rows);
                }
        }
}

