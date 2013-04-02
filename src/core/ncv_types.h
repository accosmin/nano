#ifndef NANOCV_TYPES_H
#define NANOCV_TYPES_H

#include <functional>
#include <string>
#include <vector>
#include <map>
#include <eigen3/Eigen/Core>
#include <boost/serialization/vector.hpp>

namespace ncv
{
        // Vector
        template
        <
                typename tvalue
        >
        struct vector
        {
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

        // Matrix
        template
        <
                typename tvalue
        >
        struct matrix
        {
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

        // Numerical types
        typedef std::size_t                     size_t;
        typedef std::size_t                     index_t;
        typedef std::size_t                     count_t;
        typedef std::vector<index_t>            indices_t;
        typedef std::vector<count_t>            counts_t;

        typedef double                          scalar_t;
        typedef std::vector<double>             scalars_t;

        typedef vector<scalar_t>::vector_t      scalar_vector_t;
        typedef vector<scalar_t>::vectors_t     scalar_vectors_t;

        typedef matrix<scalar_t>::matrix_t      scalar_matrix_t;
        typedef matrix<scalar_t>::matrices_t    scalar_matrices_t;

        // Strings
        typedef std::string                     string_t;
        typedef std::vector<string_t>           strings_t;
        typedef std::map<string_t, string_t>    string_map_t;

        // Lambda
        using std::placeholders::_1;
        using std::placeholders::_2;
        using std::placeholders::_3;
        using std::placeholders::_4;

        // Alignment options
        enum class align : int
        {
                left,
                center,
                right
        };
}

#endif // NANOCV_TYPES_H

