#ifndef INCLUDE_riccati_SOLVER_HPP
#define INCLUDE_riccati_SOLVER_HPP

#include <riccati/arena_matrix.hpp>
#include <riccati/memory.hpp>
#include <riccati/chebyshev.hpp>
#include <riccati/utils.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <complex>
#include <cmath>
#include <vector>

namespace riccati {

namespace internal {
inline Eigen::VectorXd logspace(double start, double end, int num,
                                double base) {
  Eigen::VectorXd result(num);
  double delta = (end - start) / (num - 1);

  for (int i = 0; i < num; ++i) {
    result[i] = std::pow(base, start + i * delta);
  }

  return result;
}
}  // namespace internal
// OmegaFun / GammaFun take in a scalar and return a scalar
template <typename OmegaFun, typename GammaFun, typename Scalar_,
          typename Integral_, bool DenseOutput = false>
class SolverInfo {
 public:
  using Scalar = Scalar_;
  using Integral = Integral_;
  using complex_t = std::complex<Scalar>;
  using matrixd_t = matrix_t<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  using vectord_t = vector_t<Scalar>;
  // Frequency function
  OmegaFun omega_fun_;
  // Friction function
  GammaFun gamma_fun_;
  // Number of nodes and diff matrices
  Integral n_nodes_{0};
  // Differentiation matrices and Vectors of Chebyshev nodes
  std::vector<std::pair<matrixd_t, vectord_t>> chebyshev_;
  // Lengths of node vectors
  vectord_t ns_;
  Integral n_idx_;
  Integral p_idx_;

  /**
   * Values of the independent variable evaluated at `p` points
   * over the interval [x, x + `h`] lying in between Chebyshev nodes, where x is
   * the value of the independent variable at the start of the current step and
   * `h` is the current stepsize. The in-between points :math:`\\tilde{x}_p` are
   * defined by
   * $$
   * \\tilde{x}_p = \cos\left( \\frac{(2k + 1)\pi}{2p} \\right), \quad k = 0, 1,
   * \ldots p-1.
   * $$
   */
  vectord_t xp_interp_;

  matrixd_t L_;
  // Clenshaw-Curtis quadrature weights
  vectord_t quadwts_;

  matrixd_t integration_matrix_;
  /**
   * Minimum and maximum (number of Chebyshev nodes - 1) to use inside
   * Chebyshev collocation steps. The step will use `nmax` nodes or the
   * minimum number of nodes necessary to achieve the required local error,
   * whichever is smaller. If `nmax` > 2`nini`, collocation steps will be
   * attempted with :math:`2^i` `nini` nodes at the ith iteration.
   */
  Integral nini_;
  Integral nmax_;
  // (Number of Chebyshev nodes - 1) to use for computing Riccati steps.
  Integral n_;
  // (Number of Chebyshev nodes - 1) to use for estimating Riccati stepsizes.
  Integral p_;

  static constexpr bool denseout_{DenseOutput};  // Dense output flag
 private:
  inline auto build_chebyshev(Integral nini, Integral n_nodes) {
    std::vector<std::pair<matrixd_t, vectord_t>> res(
        n_nodes + 1, std::make_pair(matrixd_t{}, vectord_t{}));
    // Compute Chebyshev nodes and differentiation matrices
    for (Integral i = 0; i <= n_nodes_; ++i) {
      res[i] = chebyshev<Scalar>(nini * std::pow(2, i));
    }
    return res;
  }

 public:
  /**
   * @brief Construct a new Solver Info object
   * @tparam OmegaFun_ Type of the frequency function. Must be able to take in
   * and return scalars and vectors.
   * @tparam GammaFun_ Type of the friction function. Must be able to take in
   * and return scalars and vectors.
   * @param omega_fun Frequency function
   * @param gamma_fun Friction function
   * @param nini Minimum number of Chebyshev nodes to use inside Chebyshev
   * collocation steps.
   * @param nmax Maximum number of Chebyshev nodes to use inside Chebyshev
   * collocation steps.
   * @param n (Number of Chebyshev nodes - 1) to use inside Chebyshev
   * collocation steps.
   * @param p (Number of Chebyshev nodes - 1) to use for computing Riccati
   * steps.
   */
  template <typename OmegaFun_, typename GammaFun_>
  SolverInfo(OmegaFun_&& omega_fun, GammaFun_&& gamma_fun, Integral nini,
             Integral nmax, Integral n, Integral p)
      : omega_fun_(std::forward<OmegaFun_>(omega_fun)),
        gamma_fun_(std::forward<GammaFun_>(gamma_fun)),
        n_nodes_(log2(nmax / nini) + 1),
        chebyshev_(build_chebyshev(nini, n_nodes_)),
        ns_(internal::logspace(std::log2(nini), std::log2(nini) + n_nodes_ - 1,
                               n_nodes_, 2.0)),
        n_idx_(
            std::distance(ns_.begin(), std::find(ns_.begin(), ns_.end(), n))),
        p_idx_(
            std::distance(ns_.begin(), std::find(ns_.begin(), ns_.end(), p))),
        xp_interp_((vector_t<Scalar>::LinSpaced(
                        p, pi<Scalar>() / (2.0 * p),
                        pi<Scalar>() * (1.0 - (1.0 / (2.0 * p))))
                        .array())
                       .cos()
                       .matrix()),
        L_(interpolate(this->xp(), xp_interp_, dummy_allocator{})),
        quadwts_(quad_weights<Scalar>(n)),
        integration_matrix_(DenseOutput ? integration_matrix<Scalar>(n + 1)
                                        : matrixd_t(0, 0)),
        nini_(nini),
        nmax_(nmax),
        n_(n),
        p_(p) {}

  RICCATI_ALWAYS_INLINE const auto& Dn() const noexcept {
    return chebyshev_[n_idx_].first;
  }
  RICCATI_ALWAYS_INLINE const auto& Dn(std::size_t idx) const noexcept {
    return chebyshev_[idx].first;
  }

  /**
   * Values of the independent variable evaluated at (`n` + 1) Chebyshev
   * nodes over the interval [x, x + `h`], where x is the value of the
   * independent variable at the start of the current step and `h` is the
   * current stepsize.
   */
  RICCATI_ALWAYS_INLINE const auto& xn() const noexcept {
    return chebyshev_[n_idx_].second;
  }

  /**
   * Values of the independent variable evaluated at (`p` + 1) Chebyshev
   * nodes over the interval [x, x + `h`], where x is the value of the
   * independent variable at the start of the current step and `h` is the
   * current stepsize.
   */
  RICCATI_ALWAYS_INLINE const auto& xp() const noexcept {
    return chebyshev_[p_idx_].second;
  }

  RICCATI_ALWAYS_INLINE const auto& xp_interp() const noexcept {
    return xp_interp_;
  }

  /**
   * Interpolation matrix of size (`p`+1, `p`), used for interpolating
   * function between the nodes `xp` and `xpinterp` (for computing Riccati
   * stepsizes).
   */
  RICCATI_ALWAYS_INLINE const auto& L() const noexcept { return L_; }
};

/**
 * @brief Construct a new Solver Info object
 * @tparam DenseOutput Flag to enable dense output
 * @tparam Scalar A scalar type used for calculations
 * @tparam OmegaFun Type of the frequency function. Must be able to take in and
 *  return scalars and vectors.
 * @tparam GammaFun Type of the friction function. Must be able to take in and
 *  return scalars and vectors.
 * @tparam Integral An integral type
 * @param omega_fun Frequency function
 * @param gamma_fun Friction function
 * @param nini Minimum number of Chebyshev nodes to use inside Chebyshev
 * collocation steps.
 * @param nmax Maximum number of Chebyshev nodes to use inside Chebyshev
 * collocation steps.
 * @param n (Number of Chebyshev nodes - 1) to use inside Chebyshev
 * collocation steps.
 * @param p (Number of Chebyshev nodes - 1) to use for computing Riccati
 * steps.
 */
template <bool DenseOutput, typename Scalar, typename OmegaFun,
          typename GammaFun, typename Integral>
inline auto make_solver(OmegaFun&& omega_fun, GammaFun&& gamma_fun,
                        Integral nini, Integral nmax, Integral n, Integral p) {
  if (n < nini) {
    throw std::invalid_argument("n must be greater than or equal to nini");
  }
  if (n > nmax) {
    throw std::invalid_argument("n must be less than or equal to nmax");
  }
  if (p < nini) {
    throw std::invalid_argument("p must be greater than or equal to nini");
  }
  if (p > nmax) {
    throw std::invalid_argument("p must be less than or equal to nmax");
  }
  return SolverInfo<std::decay_t<OmegaFun>, std::decay_t<GammaFun>, Scalar,
                    Integral, DenseOutput>(std::forward<OmegaFun>(omega_fun),
                                           std::forward<GammaFun>(gamma_fun),
                                           nini, nmax, n, p);
}

}  // namespace riccati

#endif
