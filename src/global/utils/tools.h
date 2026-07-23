/**
 * @file utils/tools.h
 * @brief Helper functions for general use
 * @implements
 *   - tools::ArrayImbalance -> unsigned short
 *   - tools::TensorProduct<> -> boundaries_t<T>
 *   - tools::decompose1D -> std::vector<ncells_t>
 *   - tools::divideInProportions2D -> std::tuple<unsigned int, unsigned int>
 *   - tools::divideInProportions3D -> std::tuple<unsigned int, unsigned int, unsigned int>
 *   - tools::Decompose -> std::vector<std::vector<ncells_t>>
 *   - tools::Tracker
 * @namespaces:
 *   - tools::
 */

#ifndef UTILS_TOOLS_H
#define UTILS_TOOLS_H

#include "global.h"

#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <chrono>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

namespace tools {

  /**
   * @brief Compute the imbalance of a list of nonnegative values
   * @param values List of values
   * @return Imbalance of the list (0...100)
   */
  template <typename T>
  auto ArrayImbalance(const std::vector<T>& values) -> unsigned short {
    raise::ErrorIf(values.empty(), "Disbalance error: value array is empty", HERE);
    const auto mean = static_cast<double>(std::accumulate(values.begin(),
                                                          values.end(),
                                                          static_cast<T>(0))) /
                      static_cast<double>(values.size());
    const auto sq_sum = static_cast<double>(std::inner_product(values.begin(),
                                                               values.end(),
                                                               values.begin(),
                                                               static_cast<T>(0)));
    if (cmp::AlmostZero_host(sq_sum) || cmp::AlmostZero_host(mean)) {
      return 0;
    }
    const auto cv = std::sqrt(
      sq_sum / static_cast<double>(values.size()) / mean - 1.0);
    return static_cast<unsigned short>(100.0 / (1.0 + math::exp(-cv)));
  }

  /**
   * @brief Compute a tensor product of a list of vectors
   * @param list List of vectors
   * @return Tensor product of list
   */
  template <typename T>
  inline auto TensorProduct(const std::vector<std::vector<T>>& list)
    -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> result = { {} };
    for (const auto& sublist : list) {
      std::vector<std::vector<T>> temp;
      for (const auto& element : sublist) {
        for (const auto& r : result) {
          temp.push_back(r);
          temp.back().push_back(element);
        }
      }
      result = temp;
    }
    return result;
  }

  /**
   * @brief Decompose a 1D domain into ndomains domains roughly equally
   * @param ndomains Number of domains
   * @param ncells Number of cells
   */
  inline auto decompose1D(unsigned int ndomains, ncells_t ncells)
    -> std::vector<ncells_t> {
    auto size          = (ncells_t)((double)ncells / (double)ndomains);
    auto ncells_domain = std::vector<ncells_t>(ndomains, size);
    for (auto i { 0u }; i < ncells - size * ndomains; ++i) {
      ncells_domain[i] += 1;
    }
    auto sum = std::accumulate(ncells_domain.begin(),
                               ncells_domain.end(),
                               (ncells_t)0);
    raise::ErrorIf(sum != ncells, "Decomposition error: sum != ncells", HERE);
    raise::ErrorIf(ncells_domain.size() != (std::size_t)ndomains,
                   "Decomposition error: size != ndomains",
                   HERE);
    for (unsigned int d = 0; d < ndomains; ++d) {
      raise::ErrorIf(ncells_domain[d] < 5, "ncells < 5", HERE);
    }
    return ncells_domain;
  }

  /**
   * @brief Decompose a 1D domain into ndomains contiguous blocks of roughly
   *        equal cumulative weight (static load balancing).
   * @param ndomains Number of domains
   * @param weights  Per-cell weight; length == number of cells.  Block
   *                 boundaries are chosen so that the sum of weights per block
   *                 is as equal as possible, subject to >= 5 cells per block.
   * @note With a uniform weight this reduces to decompose1D.  Used e.g. to put
   *       more (narrower) blocks where the particle load is high.
   */
  inline auto weightedDecompose1D(unsigned int               ndomains,
                                  const std::vector<real_t>& weights)
    -> std::vector<ncells_t> {
    const ncells_t ncells = (ncells_t)weights.size();
    raise::ErrorIf(ncells < 5 * ndomains,
                   "weightedDecompose1D: need >= 5 cells per domain",
                   HERE);
    // cumulative weight, cum[i] = sum of weights[0..i-1]
    std::vector<double> cum(ncells + 1, 0.0);
    for (ncells_t i { 0 }; i < ncells; ++i) {
      raise::ErrorIf(weights[i] < ZERO, "weightedDecompose1D: negative weight", HERE);
      cum[i + 1] = cum[i] + (double)weights[i];
    }
    const double total = cum[ncells];
    raise::ErrorIf(total <= 0.0, "weightedDecompose1D: non-positive total weight", HERE);

    std::vector<ncells_t> ncells_domain(ndomains, 0);
    ncells_t              start = 0;
    for (unsigned int d { 0 }; d < ndomains; ++d) {
      if (d == ndomains - 1) {
        ncells_domain[d] = ncells - start; // last block takes the remainder
        break;
      }
      // target cumulative weight at this block's right boundary
      const double target = total * (double)(d + 1) / (double)ndomains;
      // >= 5 cells here, and leave >= 5 cells for each remaining block
      const ncells_t min_end = start + 5;
      const ncells_t max_end = ncells - 5 * (ndomains - d - 1);
      ncells_t       end     = min_end;
      while (end < max_end and
             std::fabs(cum[end + 1] - target) < std::fabs(cum[end] - target)) {
        ++end;
      }
      ncells_domain[d] = end - start;
      start            = end;
    }
    const auto sum = std::accumulate(ncells_domain.begin(),
                                     ncells_domain.end(),
                                     (ncells_t)0);
    raise::ErrorIf(sum != ncells, "weightedDecompose1D: sum != ncells", HERE);
    for (unsigned int d { 0 }; d < ndomains; ++d) {
      raise::ErrorIf(ncells_domain[d] < 5, "weightedDecompose1D: ncells < 5", HERE);
    }
    return ncells_domain;
  }

  /**
   * @brief Distribute a 2D domain into ntot domains with rough proportions s1 and s2
   * @param ntot Number of domains
   * @param s1 Proportion of the first dimension
   * @param s2 Proportion of the second dimension
   */
  inline auto divideInProportions2D(unsigned int ntot, unsigned int s1, unsigned int s2)
    -> std::tuple<unsigned int, unsigned int> {
    auto n1 = (unsigned int)(std::sqrt((double)ntot * (double)s1 / (double)s2));
    if (n1 == 0) {
      return { 1, ntot };
    } else if (n1 > ntot) {
      return { ntot, 1 };
    } else {
      while (ntot % n1 != 0) {
        n1++;
        raise::ErrorIf(n1 > ntot, "Decomposition2D error: n1 > ntot", HERE);
      }
      return { n1, ntot / n1 };
    }
  }

  /**
   * @brief Distribute a 3D domain into ntot domains with rough proportions s1, s2 and s3
   * @param ntot Number of domains
   * @param s1 Proportion of the first dimension
   * @param s2 Proportion of the second dimension
   * @param s3 Proportion of the third dimension
   */
  inline auto divideInProportions3D(unsigned int ntot,
                                    unsigned int s1,
                                    unsigned int s2,
                                    unsigned int s3)
    -> std::tuple<unsigned int, unsigned int, unsigned int> {
    auto n1 = (unsigned int)(std::cbrt(
      (double)ntot * (double)(SQR(s1)) / (double)(s2 * s3)));
    if (n1 > ntot) {
      return { ntot, 1, 1 };
    } else {
      if (n1 == 0) {
        n1 = 1;
      }
      while (ntot % n1 != 0) {
        n1++;
        raise::ErrorIf(n1 > ntot, "Decomposition3D error: n1 > ntot", HERE);
      }
      auto [n2, n3] = divideInProportions2D(ntot / n1, s2, s3);
      return { n1, n2, n3 };
    }
  }

  /**
   * @brief Decompose a domain into ndomains domains
   * @param ndomains Number of domains
   * @param ncells Number of cells in each dimension
   * @param decomposition Number of domains in each dimension
   *
   * @return A vector of vectors containing the number of cells in each domain
   * in each dimension
   *
   * @note If decomposition has -1, it will be calculated automatically
   */
  inline auto Decompose(unsigned int                 ndomains,
                        const std::vector<ncells_t>& ncells,
                        const std::vector<int>&      decomposition,
                        const std::vector<std::vector<real_t>>& weights = {})
    -> std::vector<std::vector<ncells_t>> {
    const auto dimension = ncells.size();
    raise::ErrorIf(dimension != decomposition.size(),
                   "Decomposition error: dimension != decomposition.size",
                   HERE);
    // split one dimension into `nd` blocks: weighted if a per-cell weight
    // vector is supplied for that dimension, else equal cells.
    const auto split = [&](unsigned int nd, std::size_t dim_idx) {
      if (dim_idx < weights.size() and not weights[dim_idx].empty()) {
        return weightedDecompose1D(nd, weights[dim_idx]);
      }
      return decompose1D(nd, ncells[dim_idx]);
    };
    if (dimension == 1) {
      /* 1D ----------------------------------------------------------------- */
      return { split(ndomains, 0) };
    } else if (dimension == 2) {
      /* 2D ----------------------------------------------------------------- */
      unsigned int n1 { 0 }, n2 { 0 };
      if (decomposition[0] > 0 && decomposition[1] > 0) {
        n1 = decomposition[0];
        n2 = decomposition[1];
      } else if (decomposition[0] > 0 && decomposition[1] < 0) {
        n1 = decomposition[0];
        raise::ErrorIf(ndomains % n1 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n2 = ndomains / n1;
      } else if (decomposition[0] < 0 && decomposition[1] > 0) {
        n2 = decomposition[1];
        raise::ErrorIf(ndomains % n2 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n1 = ndomains / n2;
      } else if (decomposition[0] < 0 && decomposition[1] < 0) {
        std::tie(n1, n2) = divideInProportions2D(ndomains, ncells[0], ncells[1]);
      } else {
        raise::Error("Decomposition error: invalid decomposition", HERE);
      }
      raise::ErrorIf(n1 * n2 != ndomains,
                     "Decomposition error: n1 * n2 != ndomains",
                     HERE);
      return { split(n1, 0), split(n2, 1) };
    } else {
      /* 3D ----------------------------------------------------------------- */
      unsigned int n1 { 0 }, n2 { 0 }, n3 { 0 };
      if (decomposition[0] > 0 && decomposition[1] > 0 && decomposition[2] > 0) {
        n1 = decomposition[0];
        n2 = decomposition[1];
        n3 = decomposition[2];
      } else if (decomposition[0] < 0 && decomposition[1] > 0 &&
                 decomposition[2] > 0) {
        n2 = decomposition[1];
        n3 = decomposition[2];
        raise::ErrorIf(ndomains % (n2 * n3) != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n1 = ndomains / (n2 * n3);
      } else if (decomposition[0] > 0 && decomposition[1] < 0 &&
                 decomposition[2] > 0) {
        n1 = decomposition[0];
        n3 = decomposition[2];
        raise::ErrorIf(ndomains % (n1 * n3) != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n2 = ndomains / (n1 * n3);
      } else if (decomposition[0] > 0 && decomposition[1] > 0 &&
                 decomposition[2] < 0) {
        n1 = decomposition[0];
        n2 = decomposition[1];
        raise::ErrorIf(ndomains % (n1 * n2) != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n3 = ndomains / (n1 * n2);
      } else if (decomposition[0] < 0 && decomposition[1] < 0 &&
                 decomposition[2] > 0) {
        n3 = decomposition[2];
        raise::ErrorIf(ndomains % n3 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        std::tie(n1,
                 n2) = divideInProportions2D(ndomains / n3, ncells[0], ncells[1]);
      } else if (decomposition[0] < 0 && decomposition[1] > 0 &&
                 decomposition[2] < 0) {
        n2 = decomposition[1];
        raise::ErrorIf(ndomains % n2 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        std::tie(n1,
                 n3) = divideInProportions2D(ndomains / n2, ncells[0], ncells[2]);
      } else if (decomposition[0] > 0 && decomposition[1] < 0 &&
                 decomposition[2] < 0) {
        n1 = decomposition[0];
        raise::ErrorIf(ndomains % n1 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        std::tie(n2,
                 n3) = divideInProportions2D(ndomains / n1, ncells[1], ncells[2]);
      } else if (decomposition[0] < 0 && decomposition[1] < 0 &&
                 decomposition[2] < 0) {
        std::tie(n1, n2, n3) = divideInProportions3D(ndomains,
                                                     ncells[0],
                                                     ncells[1],
                                                     ncells[2]);
      }
      raise::ErrorIf(n1 * n2 * n3 != ndomains,
                     "Decomposition error: n1 * n2 * n3 != ndomains",
                     HERE);
      return { split(n1, 0), split(n2, 1), split(n3, 2) };
    }
  }

  /**
   * Class for tracking the passage of time either in steps, physical time units, or walltime
   *
   * @note Primarily used for writing checkpoints and all types of outputs at specified intervals
   */
  class Tracker {
    bool m_initialized { false };

    std::string m_type;
    timestep_t  m_interval { 0u };
    simtime_t   m_interval_time { -1.0 };
    bool        m_use_time { false };

    timestamp_t m_start_walltime;
    timestamp_t m_end_walltime;
    bool        m_walltime_pending { false };

    simtime_t m_last_output_time { -1.0 };

  public:
    Tracker() = default;

    Tracker(const std::string& type,
            timestep_t         interval,
            simtime_t          interval_time,
            const std::string& end_walltime = "",
            const timestamp_t& start_walltime = std::chrono::system_clock::now()) {
      init(type, interval, interval_time, end_walltime, start_walltime);
    }

    ~Tracker() = default;

    void init(const std::string& type,
              timestep_t         interval,
              simtime_t          interval_time,
              const std::string& end_walltime = "",
              const timestamp_t& start_walltime = std::chrono::system_clock::now()) {
      m_initialized    = true;
      m_type           = type;
      m_interval       = interval;
      m_interval_time  = interval_time;
      m_use_time       = interval_time > 0.0;
      m_start_walltime = start_walltime;
      if (not(end_walltime.empty() or end_walltime == "00:00:00")) {
        m_walltime_pending = true;
        raise::ErrorIf(end_walltime.size() != 8,
                       "invalid end walltime format, expected HH:MM:SS",
                       HERE);
        m_end_walltime = m_start_walltime +
                         std::chrono::hours(std::stoi(end_walltime.substr(0, 2))) +
                         std::chrono::minutes(std::stoi(end_walltime.substr(3, 2))) +
                         std::chrono::seconds(std::stoi(end_walltime.substr(6, 2)));
      }
    }

    auto shouldWrite(timestep_t step, simtime_t time) -> bool {
      raise::ErrorIf(!m_initialized, "Tracker not initialized", HERE);
      if (m_walltime_pending and
          (std::chrono::system_clock::now() > m_end_walltime)) {
        m_walltime_pending = false;
        return true;
      } else if (m_use_time) {
        if ((m_last_output_time < 0) or
            (time - m_last_output_time >= m_interval_time)) {
          m_last_output_time = time;
          return true;
        } else {
          return false;
        }
      } else {
        return step % m_interval == 0;
      }
    }
  };

} // namespace tools

#endif // UTILS_TOOLS_H
