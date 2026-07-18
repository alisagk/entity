/**
 * @file archetypes/energy_dist.h
 * @brief Defines an archetype for energy distributions
 * @implements
 *   - arch::energy_dist::Cold<>
 *   - arch::energy_dist::Powerlaw<>
 *   - arch::energy_dist::Maxwellian<>
 * @namespaces:
 *   - arch::energy_dist::
 */

#ifndef ARCHETYPES_ENERGY_DIST_HPP
#define ARCHETYPES_ENERGY_DIST_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cmath>

namespace arch::energy_dist {
  using namespace ntt;

  template <Dimension D>
  struct Cold {
    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {

      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }
  };

  template <Dimension D>
  struct Powerlaw {

    Powerlaw(random_number_pool_t& pool, real_t g_min, real_t g_max, real_t pl_ind)
      : g_min { g_min }
      , g_max { g_max }
      , pl_ind { pl_ind }
      , pool { pool } {}

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {
      auto rand_gen = pool.get_state();
      auto rand_X1  = Random<real_t>(rand_gen);
      auto rand_gam = ONE;

      // Power-law distribution from uniform (see https://mathworld.wolfram.com/RandomNumber.html)
      if (pl_ind != -ONE) {
        rand_gam += math::pow(
          math::pow(g_min, ONE + pl_ind) +
            (-math::pow(g_min, ONE + pl_ind) + math::pow(g_max, ONE + pl_ind)) *
              rand_X1,
          ONE / (ONE + pl_ind));
      } else {
        rand_gam += math::pow(g_min, ONE - rand_X1) * math::pow(g_max, rand_X1);
      }
      auto rand_u  = math::sqrt(SQR(rand_gam) - ONE);
      auto rand_X2 = Random<real_t>(rand_gen);
      auto rand_X3 = Random<real_t>(rand_gen);
      v[0]         = rand_u * (TWO * rand_X2 - ONE);
      v[2]         = TWO * rand_u * math::sqrt(rand_X2 * (ONE - rand_X2));
      v[1] = v[2] * math::cos(static_cast<real_t>(constant::TWO_PI) * rand_X3);
      v[2] = v[2] * math::sin(static_cast<real_t>(constant::TWO_PI) * rand_X3);

      pool.free_state(rand_gen);
    }

  private:
    const real_t         g_min, g_max, pl_ind;
    random_number_pool_t pool;
  };

  Inline void JuttnerSinge(vec_t<Dim::_3D>&            v,
                           real_t                      temp,
                           const random_number_pool_t& pool) {
    auto   rand_gen = pool.get_state();
    real_t randX1, randX2;
    if (temp < static_cast<real_t>(0.1)) {
      // Juttner-Synge distribution using the Box-Muller method - non-relativistic
      randX1 = Random<real_t>(rand_gen);
      while (cmp::AlmostZero(randX1)) {
        randX1 = Random<real_t>(rand_gen);
      }
      randX1 = math::sqrt(-TWO * math::log(randX1));
      randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
      v[0]   = randX1 * math::cos(randX2) * math::sqrt(temp);

      randX1 = Random<real_t>(rand_gen);
      while (cmp::AlmostZero(randX1)) {
        randX1 = Random<real_t>(rand_gen);
      }
      randX1 = math::sqrt(-TWO * math::log(randX1));
      randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
      v[1]   = randX1 * math::cos(randX2) * math::sqrt(temp);

      randX1 = Random<real_t>(rand_gen);
      while (cmp::AlmostZero(randX1)) {
        randX1 = Random<real_t>(rand_gen);
      }
      randX1 = math::sqrt(-TWO * math::log(randX1));
      randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
      v[2]   = randX1 * math::cos(randX2) * math::sqrt(temp);
    } else {
      // Juttner-Synge distribution using the Sobol method - relativistic
      auto randu   = ONE;
      auto randeta = Random<real_t>(rand_gen);
      while (SQR(randeta) <= SQR(randu) + ONE) {
        randX1 = Random<real_t>(rand_gen) * Random<real_t>(rand_gen) *
                 Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX1)) {
          randX1 = Random<real_t>(rand_gen) * Random<real_t>(rand_gen) *
                   Random<real_t>(rand_gen);
        }
        randu  = -temp * math::log(randX1);
        randX2 = Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX2)) {
          randX2 = Random<real_t>(rand_gen);
        }
        randeta = -temp * math::log(randX1 * randX2);
      }
      randX1 = Random<real_t>(rand_gen);
      randX2 = Random<real_t>(rand_gen);
      v[0]   = randu * (TWO * randX1 - ONE);
      v[2]   = TWO * randu * math::sqrt(randX1 * (ONE - randX1));
      v[1]   = v[2] * math::cos(static_cast<real_t>(constant::TWO_PI) * randX2);
      v[2]   = v[2] * math::sin(static_cast<real_t>(constant::TWO_PI) * randX2);
    }
    pool.free_state(rand_gen);
  }

  template <bool CanBoost>
  Inline void SampleFromMaxwellian(vec_t<Dim::_3D>&            v,
                                   const random_number_pool_t& pool,
                                   real_t                      temperature,
                                   real_t boost_velocity = static_cast<real_t>(0),
                                   in   boost_direction = in::x1,
                                   bool flip_velocity   = false) {
    if (cmp::AlmostZero(temperature)) {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    } else {
      JuttnerSinge(v, temperature, pool);
    }
    if constexpr (CanBoost) {
      // Boost a symmetric distribution to a relativistic speed using flipping
      // method https://arxiv.org/pdf/1504.03910.pdf
      // @note: boost only when using cartesian coordinates
      if (not cmp::AlmostZero(boost_velocity)) {
        const auto boost_dir = static_cast<dim_t>(boost_direction);
        const auto boost_beta { boost_velocity /
                                math::sqrt(ONE + SQR(boost_velocity)) };
        const auto gamma { U2GAMMA(v[0], v[1], v[2]) };
        auto       rand_gen = pool.get_state();
        if (-boost_beta * v[boost_dir] > gamma * Random<real_t>(rand_gen)) {
          v[boost_dir] = -v[boost_dir];
        }
        pool.free_state(rand_gen);
        v[boost_dir] = math::sqrt(ONE + SQR(boost_velocity)) *
                       (v[boost_dir] + boost_beta * gamma);
        if (flip_velocity) {
          v[0] = -v[0];
          v[1] = -v[1];
          v[2] = -v[2];
        }
      }
    }
  }

  template <Dimension D, Coord::type C>
  struct Maxwellian {
    Maxwellian(random_number_pool_t&      pool,
               real_t                     temperature,
               const std::vector<real_t>& drift_four_vel = { ZERO, ZERO, ZERO })
      : pool { pool }
      , temperature { temperature } {
      raise::ErrorIf(drift_four_vel.size() != 3,
                     "Maxwellian: Drift velocity must be a 3D vector",
                     HERE);
      raise::ErrorIf(temperature < ZERO,
                     "Maxwellian: Temperature must be non-negative",
                     HERE);
      if constexpr (C == Coord::Cartesian) {
        drift_4vel = NORM(drift_four_vel[0], drift_four_vel[1], drift_four_vel[2]);
        if (cmp::AlmostZero_host(drift_4vel)) {
          drift_dir = 0;
        } else {
          drift_3vel   = drift_4vel / math::sqrt(ONE + SQR(drift_4vel));
          drift_dir_x1 = drift_four_vel[0] / drift_4vel;
          drift_dir_x2 = drift_four_vel[1] / drift_4vel;
          drift_dir_x3 = drift_four_vel[2] / drift_4vel;

          // assume drift is in an arbitrary direction
          drift_dir = 4;
          // check whether drift is in one of principal directions
          for (auto d { 0u }; d < 3u; ++d) {
            const auto dprev = (d + 2) % 3;
            const auto dnext = (d + 1) % 3;
            if (cmp::AlmostZero_host(drift_four_vel[dprev]) and
                cmp::AlmostZero_host(drift_four_vel[dnext])) {
              drift_dir = SIGN(drift_four_vel[d]) * (static_cast<real_t>(d + 1));
              break;
            }
          }
        }
        raise::ErrorIf(drift_dir > 3 and drift_dir != 4,
                       "Maxwellian: Incorrect drift direction",
                       HERE);
        raise::ErrorIf(
          drift_dir != 0 and (C != Coord::Cartesian),
          "Maxwellian: Boosting is only supported in Cartesian coordinates",
          HERE);
      }
    }

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {
      if (cmp::AlmostZero(temperature)) {
        v[0] = ZERO;
        v[1] = ZERO;
        v[2] = ZERO;
      } else {
        JuttnerSinge(v, temperature, pool);
      }
      // @note: boost only when using cartesian coordinates
      if constexpr (C == Coord::Cartesian) {
        if (drift_dir != 0) {
          // Boost an isotropic Maxwellian with a drift velocity using
          // flipping method https://arxiv.org/pdf/1504.03910.pdf
          // 1. apply drift in X1 direction
          const auto gamma { U2GAMMA(v[0], v[1], v[2]) };
          auto       rand_gen = pool.get_state();
          if (-drift_3vel * v[0] > gamma * Random<real_t>(rand_gen)) {
            v[0] = -v[0];
          }
          pool.free_state(rand_gen);
          v[0] = math::sqrt(ONE + SQR(drift_4vel)) * (v[0] + drift_3vel * gamma);
          // 2. rotate to desired orientation
          if (drift_dir == -1) {
            v[0] = -v[0];
          } else if (drift_dir == 2 || drift_dir == -2) {
            const auto tmp = v[1];
            v[1]           = drift_dir > 0 ? v[0] : -v[0];
            v[0]           = tmp;
          } else if (drift_dir == 3 || drift_dir == -3) {
            const auto tmp = v[2];
            v[2]           = drift_dir > 0 ? v[0] : -v[0];
            v[0]           = tmp;
          } else if (drift_dir == 4) {
            vec_t<Dim::_3D> v_old;
            v_old[0] = v[0];
            v_old[1] = v[1];
            v_old[2] = v[2];

            v[0] = v_old[0] * drift_dir_x1 - v_old[1] * drift_dir_x2 -
                   v_old[2] * drift_dir_x3;
            v[1] = (v_old[0] * drift_dir_x2 * (drift_dir_x1 + ONE) +
                    v_old[1] *
                      (SQR(drift_dir_x1) + drift_dir_x1 + SQR(drift_dir_x3)) -
                    v_old[2] * drift_dir_x2 * drift_dir_x3) /
                   (drift_dir_x1 + ONE);
            v[2] = (v_old[0] * drift_dir_x3 * (drift_dir_x1 + ONE) -
                    v_old[1] * drift_dir_x2 * drift_dir_x3 -
                    v_old[2] * (-drift_dir_x1 + SQR(drift_dir_x3) - ONE)) /
                   (drift_dir_x1 + ONE);
          }
        }
      }
    }

  private:
    random_number_pool_t pool;

    const real_t temperature;

    real_t drift_3vel { ZERO }, drift_4vel { ZERO };
    // components of the unit vector in the direction of the drift
    real_t drift_dir_x1 { ZERO }, drift_dir_x2 { ZERO }, drift_dir_x3 { ZERO };

    // values of boost_dir:
    // 4 -> arbitrary direction
    // 0 -> no drift
    // +/- 1 -> +/- x1
    // +/- 2 -> +/- x2
    // +/- 3 -> +/- x3
    short drift_dir { 0 };
  };

  // -----------------------------------------------------------------------
  // FluxWeightedJuttner<D>
  //
  // Samples from f_wall(ux, uperp) ∝ (ux/γ) exp(−γ/T),  ux > 0
  // where γ = sqrt(1 + ux² + uperp²).
  //
  // This is the flux-weighted half-space distribution f_wall = vx · f_target
  // (vx = ux/γ = βx). Injecting from a wall with this distribution reproduces
  // an isotropic Jüttner f_target throughout the domain in the test-particle
  // limit.
  //
  // Algorithm: 2D inverse-transform sampling via pre-cached analytic CDFs.
  //
  //   Marginal F(ux) has analytic CDF:
  //     C_ux(ux) = T[(1+T) e^{-1/T} - (γ₀+T) e^{-γ₀/T}]   γ₀ = sqrt(1+ux²)
  //
  //   Conditional G(uperp|ux) has analytic CDF:
  //     C_c(uperp;ux) = 1 - e^{-(γ-γ₀)/T}                  γ = sqrt(1+ux²+uperp²)
  //   (The Jüttner conditional has an extra (γ+T)/(γ₀+T) factor — wrong here.)
  //
  // Both CDFs are evaluated on uniform grids in the constructor (host, double
  // precision), copied to device. On-device sampling uses binary search +
  // linear / bilinear interpolation.
  //
  // Output: v[0]=ux>0, v[1,2]=uy,uz in physical tetrad (flat-space) basis.
  // For GR injection apply transform<T,U> + normal-frame correction after.
  // -----------------------------------------------------------------------
  template <Dimension D>
  struct FluxWeightedJuttner {
    static constexpr int N_UX    = 512;
    static constexpr int N_UPERP = 512;

    FluxWeightedJuttner(random_number_pool_t& pool_, real_t temperature_)
      : pool { pool_ }
      , d_ux_grid("fwj_ux", N_UX)
      , d_uperp_grid("fwj_uperp", N_UPERP)
      , d_cdf_ux("fwj_cdf_ux", N_UX)
      , d_cdf_cond("fwj_cdf_cond", N_UX * N_UPERP) {

      const double T = static_cast<double>(temperature_);

      // Analytic (unnormalized) marginal CDF for ux; double precision avoids
      // underflow of exp(-1/T) for small T (e.g. T=0.01 → exp(-100) ≈ 3.7e-44).
      const double cdf_ux_total = T * (1.0 + T) * std::exp(-1.0 / T);

      auto cdf_ux_fn = [&](double ux) -> double {
        double g0 = std::sqrt(1.0 + ux * ux);
        return T * ((1.0 + T) * std::exp(-1.0 / T) - (g0 + T) * std::exp(-g0 / T));
      };

      // Flux-weighted conditional CDF: C_c(uperp;ux) = 1 - exp(-(γ-γ₀)/T).
      // Derivation: f(uperp|ux) ∝ (uperp/γ) exp(-(γ-γ₀)/T); with t=γ,
      // tdt=uperp d(uperp), integral gives -exp(-(t-γ₀)/T)|_{γ₀}^{γ}.
      // NOTE: the Jüttner conditional CDF has an extra (γ+T)/(γ₀+T) factor —
      // that form is wrong here because we sample from f_wall ∝ (ux/γ)·f_J.
      auto cdf_cond_fn = [&](double uperp, double ux) -> double {
        double g0 = std::sqrt(1.0 + ux * ux);
        double g  = std::sqrt(1.0 + ux * ux + uperp * uperp);
        return 1.0 - std::exp(-(g - g0) / T);
      };

      // ux_max: smallest value where C_ux(ux_max) / C_ux(∞) >= 1 - 1e-6.
      double ux_max;
      {
        double lo = 0.0, hi = std::max(1.0, 20.0 * std::sqrt(T));
        while (cdf_ux_fn(hi) / cdf_ux_total < 0.999999) {
          hi *= 2.0;
        }
        for (int k = 0; k < 100; ++k) {
          double mid = 0.5 * (lo + hi);
          (cdf_ux_fn(mid) / cdf_ux_total >= 0.999999) ? hi = mid : lo = mid;
        }
        ux_max = hi;
      }

      // uperp_max: where C_c(uperp_max; ux=0) >= 1 - 1e-6.
      // ux=0 gives the widest uperp support (γ₀ = 1, smallest possible).
      double uperp_max;
      {
        double lo = 0.0, hi = std::max(1.0, 20.0 * std::sqrt(T));
        while (cdf_cond_fn(hi, 0.0) < 0.999999) {
          hi *= 2.0;
        }
        for (int k = 0; k < 100; ++k) {
          double mid = 0.5 * (lo + hi);
          (cdf_cond_fn(mid, 0.0) >= 0.999999) ? hi = mid : lo = mid;
        }
        uperp_max = hi;
      }

      // Build host arrays.
      array_mirror_t<real_t*> h_ux("fwj_ux_h",       N_UX);
      array_mirror_t<real_t*> h_up("fwj_uperp_h",    N_UPERP);
      array_mirror_t<real_t*> h_cu("fwj_cdf_ux_h",   N_UX);
      array_mirror_t<real_t*> h_cc("fwj_cdf_cond_h", N_UX * N_UPERP);

      for (int i = 0; i < N_UX;    ++i)
        h_ux(i) = static_cast<real_t>(ux_max    * i / (N_UX    - 1));
      for (int j = 0; j < N_UPERP; ++j)
        h_up(j) = static_cast<real_t>(uperp_max * j / (N_UPERP - 1));

      // Marginal CDF normalized to [0, 1].
      for (int i = 0; i < N_UX; ++i) {
        double ux = static_cast<double>(h_ux(i));
        h_cu(i)   = static_cast<real_t>(cdf_ux_fn(ux) / cdf_ux_total);
      }
      h_cu(0)        = static_cast<real_t>(0);
      h_cu(N_UX - 1) = static_cast<real_t>(1);

      // Conditional CDF: row-major [ux_index * N_UPERP + uperp_index].
      for (int i = 0; i < N_UX; ++i) {
        double ux = static_cast<double>(h_ux(i));
        for (int j = 0; j < N_UPERP; ++j) {
          double up             = static_cast<double>(h_up(j));
          h_cc(i * N_UPERP + j) = static_cast<real_t>(cdf_cond_fn(up, ux));
        }
        h_cc(i * N_UPERP + 0)          = static_cast<real_t>(0);
        h_cc(i * N_UPERP + N_UPERP - 1) = static_cast<real_t>(1);
      }

      Kokkos::deep_copy(d_ux_grid,    h_ux);
      Kokkos::deep_copy(d_uperp_grid, h_up);
      Kokkos::deep_copy(d_cdf_ux,     h_cu);
      Kokkos::deep_copy(d_cdf_cond,   h_cc);
    }

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {
      auto rand_gen = pool.get_state();
      const real_t r1 = Random<real_t>(rand_gen);
      const real_t r2 = Random<real_t>(rand_gen);
      const real_t r3 = Random<real_t>(rand_gen);
      pool.free_state(rand_gen);

      const real_t ux    = invert_1d(d_cdf_ux, d_ux_grid, N_UX, r1);
      const real_t uperp = invert_cond(ux, r2);
      const real_t phi   = static_cast<real_t>(constant::TWO_PI) * r3;

      v[0] = ux;
      v[1] = uperp * math::cos(phi);
      v[2] = uperp * math::sin(phi);
    }

  private:
    random_number_pool_t pool;
    array_t<real_t*>     d_ux_grid;    // N_UX uniform ux values in [0, ux_max]
    array_t<real_t*>     d_uperp_grid; // N_UPERP uniform uperp values in [0, uperp_max]
    array_t<real_t*>     d_cdf_ux;     // marginal CDF, length N_UX
    array_t<real_t*>     d_cdf_cond;   // conditional CDF, N_UX * N_UPERP, row-major

    // Largest lo such that arr(lo) < val (returns 0 if val <= arr(0)).
    Inline static int bsearch(const array_t<real_t*>& arr, int N, real_t val) {
      int lo = 0, hi = N - 1;
      while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        (arr(mid) < val) ? lo = mid : hi = mid;
      }
      return lo;
    }

    // Invert a 1D monotone CDF on a grid via binary search + linear interp.
    Inline real_t invert_1d(const array_t<real_t*>& cdf,
                             const array_t<real_t*>& grid, int N,
                             real_t r) const {
      const int    lo = bsearch(cdf, N, r);
      const int    hi = lo + 1;
      const real_t dc = cdf(hi) - cdf(lo);
      const real_t t  = (dc > ZERO) ? (r - cdf(lo)) / dc : ZERO;
      return grid(lo) + t * (grid(hi) - grid(lo));
    }

    // Invert row i of the conditional CDF d_cdf_cond(i, ·).
    Inline real_t invert_row(int i, real_t r) const {
      const int off = i * N_UPERP;
      int       lo  = 0, hi = N_UPERP - 1;
      while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        (d_cdf_cond(off + mid) < r) ? lo = mid : hi = mid;
      }
      const real_t dc = d_cdf_cond(off + hi) - d_cdf_cond(off + lo);
      const real_t t  = (dc > ZERO) ? (r - d_cdf_cond(off + lo)) / dc : ZERO;
      return d_uperp_grid(lo) + t * (d_uperp_grid(hi) - d_uperp_grid(lo));
    }

    // Invert conditional CDF with bilinear interpolation in ux.
    Inline real_t invert_cond(real_t ux, real_t r) const {
      const real_t ux_max = d_ux_grid(N_UX - 1);
      real_t       ix_f   = (ux_max > ZERO)
                              ? ux / ux_max * static_cast<real_t>(N_UX - 1)
                              : ZERO;
      int ix = static_cast<int>(ix_f);
      if (ix < 0)         ix = 0;
      if (ix >= N_UX - 1) ix = N_UX - 2;
      const real_t t_ux = ix_f - static_cast<real_t>(ix);

      const real_t up0 = invert_row(ix,     r);
      const real_t up1 = invert_row(ix + 1, r);
      return up0 * (ONE - t_ux) + up1 * t_ux;
    }
  };

} // namespace arch::energy_dist

#endif // ARCHETYPES_ENERGY_DIST_HPP
