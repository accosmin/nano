#pragma once

#include <eigen3/Eigen/Core>
#include <type_traits>
#include <vector>

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
                        return tresult(data, rows);
                }
                
                ///
                /// map data to vectors
                ///
                template
                <
                        typename tvalue_,
                        typename tsize,
                        typename tvalue = typename std::remove_const<tvalue_>::type,
                        typename tresult = Eigen::Map<const typename vector_types_t<tvalue>::tvector>
                >
                tresult make_vector(const tvalue_* data, tsize rows)
                {
                        return tresult(data, rows);
                }
        }
}

