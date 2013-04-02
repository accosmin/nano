#ifndef NANOCV_STRING_H
#define NANOCV_STRING_H

#include "ncv_types.h"
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

namespace ncv
{
        namespace text
        {
                using namespace boost::algorithm;

                // trim a string
                string_t trim(const string_t& str, const char* trim_chars = " \n\t\r");

                // split a string given some separator characters
                strings_t tokenize(const string_t& str, const char* delim_chars = " \n\t\r");

                // align a string to fill the given size
                string_t resize(const string_t& str, size_t size, align alignment = align::left);

                // compact a list of strings using the given glue string
                string_t concatenate(const strings_t& strs, const string_t& glue = ",");

                // to lower & upper string
                string_t to_lower(const string_t& str);
                string_t to_upper(const string_t& str);

                // string cast for built-in types
                template
                <
                        typename tvalue
                >
                string_t to_string(tvalue value)
                {
                        return std::to_string(value);
                }

                template
                <
                        typename tvalue
                >
                tvalue from_string(const string_t& str)
                {
                        return boost::lexical_cast<tvalue>(str);
                }

                // string cast for vectors
                // FIXME: can write it like to_string ?!
                template
                <
                        typename tvalue
                >
                string_t vto_string(const typename vector<tvalue>::vector_t& v)
                {
                        const index_t size = static_cast<index_t>(v.size());

                        string_t ret;
                        for (index_t i = 0; i < size; i ++)
                        {
                                ret += to_string<tvalue>(v(i));
                                if (i + 1 != size)
                                {
                                        ret += ", ";
                                }
                        }

                        return ret;
                }

                // decode parameter by name: [name1=value1[,name2=value2[...]]
                // the default value is returned if the parameter cannot be found or is invalid.
                template
                <
                        class tvalue
                >
                tvalue from_params(const string_t& params, const string_t& param_name, tvalue default_value)
                {
                        const strings_t tokens = tokenize(params, ",");
                        const size_t tsize = tokens.size();

                        for (size_t i = 0; i < tsize; i ++)
                        {
                                const strings_t dual = tokenize(tokens[i], "=");
                                if (dual.size() == 2 && dual[0] == param_name)
                                {
                                        string_t value = dual[1];
                                        for (   size_t j = i + 1;
                                                j < tsize && tokens[j].find("=") == string_t::npos;
                                                j ++)
                                        {
                                                value += "," + tokens[j];
                                        }

                                        try
                                        {
                                                return from_string<tvalue>(value);
                                        }
                                        catch (std::exception&)
                                        {
                                                return default_value;
                                        }
                                }
                        }

                        return default_value;
                }

                // tabulate the given columns: [header, column values, alignment, column ending]+
                template
                <
                        class... tcolumns
                >
                string_t tabulate(tcolumns... data);

                namespace impl
                {
                        // concatenate two columns
                        strings_t tabulate_concatenate(const strings_t& col1, const strings_t& col2);

                        // tabulate the given columns: [header, column values, alignment, column ending]+
                        strings_t tabulate_column(
                                const string_t& header, const strings_t& values, align alignment, const string_t& end);

                        template
                        <
                                class... tcolumns
                        >
                        strings_t tabulate_column(
                                const string_t& header, const strings_t& values, align alignment, const string_t& end,
                                tcolumns... others)
                        {
                                return  tabulate_concatenate(
                                        tabulate_column(header, values, alignment, end),
                                        tabulate_column(others...));
                        }
                }

                // tabulate the given columns: [header, column values, alignment, column ending]+
                template
                <
                        class... tcolumns
                >
                string_t tabulate(tcolumns... data)
                {
                        return concatenate(impl::tabulate_column(data...), "\n");
                }
        }
}

#endif // NANOCV_STRING_H

