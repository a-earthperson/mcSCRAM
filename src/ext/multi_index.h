/*
 * Copyright (C) 2017-2018 Olzhas Rakhimov
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/// @file
/// Helper functions to boost multi_index_container.
#pragma once

#include <boost/multi_index_container.hpp>

namespace ext {

/// Extracts (i.e., take, remove-return) a value from multi_index container.
///
/// @tparam T  The value type in the container.
/// @tparam Ts  The rest of the multi_index container type parameters.
///
/// @param[in] it  The iterator to the container.
/// @param[in,out] container  The container with the associated value.
///
/// @returns The extracted value.
///
/// @pre (it != container.end()).
///
/// @note This function breaks the contract of multi_index_container
///       by modifying the value outside of the container.
template <typename T, typename... Ts>
T extract(typename boost::multi_index_container<T, Ts...>::iterator it,
          boost::multi_index_container<T, Ts...>* container)  {
  assert(it != container->end());
  T result = std::move(const_cast<T&>(*it));  // Theft, contract-violation.
  container->erase(it);  // Requires valid iterator but not value.
  return result;
}

/// The same extraction but with an existing key-value.
template <typename T, typename... Ts>
T extract(const typename boost::multi_index_container<T, Ts...>::key_type& key,
          boost::multi_index_container<T, Ts...>* container)  {
  return extract(container->find(key), container);
}

}  // namespace ext
