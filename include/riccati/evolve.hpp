#ifndef INCLUDE_RICCATI_EVOLVE_HPP
#define INCLUDE_RICCATI_EVOLVE_HPP

#include <riccati/chebyshev.hpp>
#include <riccati/step.hpp>
#include <riccati/stepsize.hpp>
#include <riccati/utils.hpp>
#include <complex>
#include <type_traits>
#include <tuple>

namespace riccati {

/**
 * @brief Solves the differential equation y'' + 2gy' + w^2y = 0 over a given
 * interval.
 *
 * This function solves the differential equation on the interval (xi, xf),
 * starting from the initial conditions y(xi) = yi and y'(xi) = dyi. It keeps
 * the residual of the ODE below eps, and returns an interpolated solution
 * (dense output) at the points specified in x_eval.
 *
 * @tparam SolverInfo Type of the solver info object containing differentiation
 * matrices, etc.
 * @tparam Scalar Numeric scalar type, typically float or double.
 * @tparam Vec Type of the vector for dense output values, should match Scalar
 * type.
 * @tparam Vec Type of the vector for dense output values.
 * @tparam Allocator Type of the allocator for the arena memory pool.
 * @param[in] info SolverInfo object containing necessary information for the
 * solver.
 * @param[in] xi Starting value of the independent variable.
 * @param[in] xf Ending value of the independent variable.
 * @param[in] yi Initial value of the dependent variable at xi.
 * @param[in] dyi Initial derivative of the dependent variable at xi.
 * @param[in] eps Relative tolerance for the local error of both Riccati and
 * Chebyshev type steps.
 * @param[in] epsilon_h Relative tolerance for choosing the stepsize of Riccati
 * steps.
 * @param[in] init_stepsize Stepsize for the integration
 * @param[in] x_eval List of x-values where the solution is to be interpolated
 * (dense output) and returned.
 * @param[in] hard_stop If true, forces the solver to have a potentially smaller
 * last stepsize to stop exactly at xf.
 * @returns A tuple containing the following elements:
 * 0. `bool`: A boolean flag indicating whether the step was successfully taken.
 * 1. `Scalar`: The next x-value after the integration step, indicating the new
 * position in the integration domain.
 * 2. `Scalar`: The suggested next stepsize for further integration steps based
 * on the current step's data.
 * 3. `std::tuple`: The result from the @ref riccati::osc_step function which
 * includes various internal states and calculations specific to the current
 * step.
 * 4. @ref Eigen::Matrix<std::complex<Scalar>, -1, 1>: A vector containing the
 * interpolated solution at the specified `x_eval` points. This represents the
 * dense output for the current step.
 * 5. `Eigen::Index`: The starting index in the `x_eval` array corresponding to
 * the first point in the current step's dense output.
 * 6. `Eigen::Index`: The number of points in the `x_eval` array that are
 * covered by the current step's dense output.
 *
 * @note Tuple element 4 vector contains the interpolated values of the
 * differential equation's solution at the points specified by the `x_eval`
 * input parameter. These values are calculated using the dense output
 * methodology and are meant for high-accuracy interpolation between the
 * standard discrete steps of the solver.
 */
template <typename SolverInfo, typename Scalar, typename Vec>
inline auto osc_evolve(SolverInfo &&info, Scalar xi, Scalar xf,
                       std::complex<Scalar> yi, std::complex<Scalar> dyi,
                       Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                       Vec &&x_eval, bool hard_stop = false) {
  int sign = init_stepsize > 0 ? 1 : -1;
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  auto xi_scaled
      = eval(info.alloc_, scale(info.xn().array(), xi, init_stepsize).matrix());
  // Frequency and friction functions evaluated at n+1 Chebyshev nodes
  auto omega_n = eval(info.alloc_, omega(info, xi_scaled));
  auto gamma_n = eval(info.alloc_, gamma(info, xi_scaled));
  vectorc_t yeval;
  vectorc_t dyeval;
  // TODO: Add this check to regular evolve
  if (sign * (xi + init_stepsize) > sign * xf) {
    throw std::out_of_range(
        std::string("Stepsize (") + std::to_string(init_stepsize)
        + std::string(") is too large for integration range (")
        + std::to_string(xi) + std::string(", ") + std::to_string(xf)
        + std::string(")!"));
  }
  // o and g read here
  constexpr bool dense_output = compile_size_v<Vec> != 0;
  auto osc_ret = osc_step<dense_output>(info, omega_n, gamma_n, xi,
                                        init_stepsize, yi, dyi, eps);
  if (std::get<0>(osc_ret) == 0) {
    return std::make_tuple(false, xi, init_stepsize, osc_ret, vectorc_t(0),
                           vectorc_t(0), static_cast<Eigen::Index>(0),
                           static_cast<Eigen::Index>(0));
  } else {
    Eigen::Index dense_size = 0;
    Eigen::Index dense_start = 0;
    if constexpr (dense_output) {
      // Assuming x_eval is sorted we just want start and size
      std::tie(dense_start, dense_size)
          = get_slice(x_eval, sign * xi, sign * (xi + init_stepsize));
      if (dense_size != 0) {
        auto x_eval_map = x_eval.segment(dense_start, dense_size);
        auto x_eval_scaled = eval(
            info.alloc_,
            (2.0 / init_stepsize * (x_eval_map.array() - xi) - 1.0).matrix());
        auto Linterp = interpolate(info.xn(), x_eval_scaled, info.alloc_);
        auto fdense
            = eval(info.alloc_,
                   (Linterp * std::get<5>(osc_ret)).array().exp().matrix());
        yeval = std::get<7>(osc_ret).first * fdense
                + std::get<7>(osc_ret).second * fdense.conjugate();
        auto du_dense
            = eval(info.alloc_, (Linterp * std::get<6>(osc_ret))).array();
        dyeval
            = std::get<7>(osc_ret).first * (du_dense.array() * fdense.array())
              + std::get<7>(osc_ret).second
                    * (du_dense.array() * fdense.array()).conjugate();
      }
    }
    auto x_next = xi + init_stepsize;
    // o and g read here
    auto wnext = omega_n[0];
    auto gnext = gamma_n[0];
    auto dwnext = 2.0 / init_stepsize * info.Dn().row(0).dot(omega_n);
    auto dgnext = 2.0 / init_stepsize * info.Dn().row(0).dot(gamma_n);
    auto hosc_ini = sign
                    * std::min(std::min(1e8, std::abs(wnext / dwnext)),
                               std::abs(gnext / dgnext));
    if (sign * (x_next + hosc_ini) > sign * xf) {
      hosc_ini = xf - x_next;
    }
    // o and g written here
    auto h_next = choose_osc_stepsize(info, x_next, hosc_ini, epsilon_h);
    // TODO: CANNOT RETURN ARENA MATRIX FROM OSC_RET
    return std::make_tuple(true, x_next, std::get<0>(h_next), osc_ret, yeval,
                           dyeval, dense_start, dense_size);
  }
}

/**
 * @brief Solves the differential equation y'' + 2gy' + w^2y = 0 over a given
 * interval.
 *
 * This function solves the differential equation on the interval (xi, xf),
 * starting from the initial conditions y(xi) = yi and y'(xi) = dyi. It keeps
 * the residual of the ODE below eps, and returns an interpolated solution
 * (dense output) at the points specified in x_eval.
 *
 * @tparam SolverInfo Type of the solver info object containing differentiation
 * matrices, etc.
 * @tparam Scalar Numeric scalar type, typically float or double.
 * @tparam Vec Type of the vector for dense output values, should match Scalar
 * type.
 * @tparam Vec Type of the vector for dense output values.
 * @param[in] info SolverInfo object containing necessary information for the
 * solver.
 * @param[in] xi Starting value of the independent variable.
 * @param[in] xf Ending value of the independent variable.
 * @param[in] yi Initial value of the dependent variable at xi.
 * @param[in] dyi Initial derivative of the dependent variable at xi.
 * @param[in] eps Relative tolerance for the local error of both Riccati and
 * Chebyshev type steps.
 * @param[in] epsilon_h Relative tolerance for choosing the stepsize of Riccati
 * steps.
 * @param[in] init_stepsize Stepsize for the integration
 * @param[in] x_eval List of x-values where the solution is to be interpolated
 * (dense output) and returned.
 * @param[in] hard_stop If true, forces the solver to have a potentially smaller
 * last stepsize to stop exactly at xf.
 * @returns A tuple containing the following elements:
 * 0. `bool`: A boolean flag indicating whether the step was successfully taken.
 * 1. `Scalar`: The next x-value after the integration step, indicating the new
 * position in the integration domain.
 * 2. `Scalar`: The suggested next stepsize for further integration steps based
 * on the current step's data.
 * 3. `std::tuple`: The result from the @ref riccati::nonosc_step function which
 * includes various internal states and calculations specific to the current
 * step.
 * 4. @ref Eigen::Matrix<std::complex<Scalar>, -1, 1>: A vector containing the
 * interpolated solution at the specified `x_eval` points. This represents the
 * dense output for the current step.
 * 5. `Eigen::Index`: The starting index in the `x_eval` array corresponding to
 * the first point in the current step's dense output.
 * 6. `Eigen::Index`: The number of points in the `x_eval` array that are
 * covered by the current step's dense output.
 *
 * @note Tuple element 4 vector contains the interpolated values of the
 * differential equation's solution at the points specified by the `x_eval`
 * input parameter. These values are calculated using the dense output
 * methodology and are meant for high-accuracy interpolation between the
 * standard discrete steps of the solver.
 */
template <typename SolverInfo, typename Scalar, typename Vec>
inline auto nonosc_evolve(SolverInfo &&info, Scalar xi, Scalar xf,
                          std::complex<Scalar> yi, std::complex<Scalar> dyi,
                          Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                          Vec &&x_eval, bool hard_stop = false) {
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  vectorc_t yeval;
  vectorc_t dyeval;
  // TODO: Add this check to regular evolve
  const int sign = init_stepsize > 0 ? 1 : -1;
  if (sign * (xi + init_stepsize) > sign * xf) {
    throw std::out_of_range(
        std::string("Stepsize (") + std::to_string(init_stepsize)
        + std::string(") is too large for integration range (")
        + std::to_string(xi) + std::string(", ") + std::to_string(xf)
        + std::string(")!"));
  }
  auto nonosc_ret = nonosc_step(info, xi, init_stepsize, yi, dyi, eps);
  if (!std::get<0>(nonosc_ret)) {
    return std::make_tuple(false, xi, init_stepsize, nonosc_ret, vectorc_t(0),
                           vectorc_t(0), static_cast<Eigen::Index>(0),
                           static_cast<Eigen::Index>(0));
  } else {
    Eigen::Index dense_size = 0;
    Eigen::Index dense_start = 0;
    if constexpr (compile_size_v<Vec> != 0) {
      // Assuming x_eval is sorted we just want start and size
      std::tie(dense_start, dense_size)
          = get_slice(x_eval, sign * xi, sign * (xi + init_stepsize));
      if (dense_size != 0) {
        auto x_eval_map = x_eval.segment(dense_start, dense_size);

        auto xi_scaled
            = (xi + init_stepsize / 2
               + (init_stepsize / 2)
                     * std::get<2>(info.chebyshev_[std::get<6>(nonosc_ret)])
                           .array())
                  .matrix()
                  .eval();
        auto Linterp = interpolate(xi_scaled, x_eval_map, info.alloc_);
        yeval = Linterp * std::get<4>(nonosc_ret);
        dyeval = Linterp * std::get<5>(nonosc_ret);
      }
    }
    auto x_next = xi + init_stepsize;
    auto wnext = omega(info, xi + init_stepsize);
    auto hslo_ini = sign * std::min(1e8, std::abs(1.0 / wnext));
    if (sign * (x_next + hslo_ini) > sign * xf) {
      hslo_ini = xf - x_next;
    }
    auto h_next = choose_nonosc_stepsize(info, x_next, hslo_ini, epsilon_h);
    return std::make_tuple(true, x_next, h_next, nonosc_ret, yeval, dyeval,
                           dense_start, dense_size);
  }
}

/**
 * @brief Solves the differential equation y'' + 2gy' + w^2y = 0 over a given
 * interval.
 *
 * This function solves the differential equation on the interval (xi, xf),
 * starting from the initial conditions y(xi) = yi and y'(xi) = dyi. It keeps
 * the residual of the ODE below eps, and returns an interpolated solution
 * (dense output) at the points specified in x_eval.
 *
 * @tparam SolverInfo Type of the solver info object containing differentiation
 * matrices, etc.
 * @tparam Scalar Numeric scalar type, typically float or double.
 * @tparam Vec Type of the vector for dense output values, should match Scalar
 * type.
 *
 * @tparam SolverInfo Type of the solver info object containing differentiation
 * matrices, etc.
 * @tparam Scalar Numeric scalar type, typically float or double.
 * @tparam Vec Type of the vector for dense output values.
 * @param[in] info SolverInfo object containing necessary information for the
 * solver.
 * @param[in] xi Starting value of the independent variable.
 * @param[in] xf Ending value of the independent variable.
 * @param[in] yi Initial value of the dependent variable at xi.
 * @param[in] dyi Initial derivative of the dependent variable at xi.
 * @param[in] eps Relative tolerance for the local error of both Riccati and
 * Chebyshev type steps.
 * @param[in] epsilon_h Relative tolerance for choosing the stepsize of Riccati
 * steps.
 * @param[in] init_stepsize initial stepsize for the integration
 * @param[in] x_eval List of x-values where the solution is to be interpolated
 * (dense output) and returned.
 * @param[in] hard_stop If true, forces the solver to have a potentially smaller
 * last stepsize to stop exactly at xf.
 * @return A tuple containing multiple elements representing the results of the
 * ODE solving process: 0. std::vector<Scalar>: A vector containing the x-values
 * at which the solution was evaluated or interpolated. These values correspond
 * to the points in the interval [xi, xf] and include the points specified in
 * x_eval if dense output was requested.
 * 1. std::vector<std::complex<Scalar>>: A vector of complex numbers
 * representing the solution y(x) of the differential equation at each x-value
 * from the corresponding vector of x-values.
 * 2. std::vector<std::complex<Scalar>>: A vector of complex numbers
 * representing the derivative of the solution, y'(x), at each x-value from the
 * corresponding vector of x-values.
 * 3. std::vector<int>: A vector indicating the success status of the solver at
 * each step. Each element corresponds to a step in the solver process, where a
 * value of 1 indicates success, and 0 indicates failure.
 * 4. std::vector<int>: A vector indicating the type of step taken at each point
 * in the solution process. Each element corresponds to a step in the solver
 * process, where a value of 1 indicates an oscillatory step and 0 indicates a
 *   non-oscillatory step.
 * 5. std::vector<Scalar>: A vector containing the phase angle at each step of
 * the solution process if relevant. This is especially applicable for
 * oscillatory solutions and may not be used for all types of differential
 * equations.
 * 6. @ref Eigen::Matrix<std::complex<Scalar>, -1, 1>: A vector containing the
 * interpolated solution at the specified `x_eval` The function returns these
 * 7. An array of logging information for the solver process. This can be added with the
 *  info() from the `SolverInfo` object to get the total log information.
 * vectors encapsulated in a standard tuple, providing comprehensive information
 * about the solution process, including where the solution was evaluated, the
 * values and derivatives of the solution at those points, success status of
 * each step, type of each step, and phase angles where applicable.
 */
template <typename SolverInfo, typename Scalar, typename Vec>
inline auto evolve(SolverInfo &info, Scalar xi, Scalar xf,
                   std::complex<Scalar> yi, std::complex<Scalar> dyi,
                   Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                   Vec &&x_eval, bool hard_stop = false,
                   LogLevel log_level = riccati::LogLevel::ERROR) {
  static_assert(std::is_floating_point<Scalar>::value,
                "Scalar type must be a floating-point type.");
  using vectord_t = vector_t<Scalar>;
  Scalar direction = init_stepsize > 0 ? 1 : -1;
  if (init_stepsize * (xf - xi) < 0) {
    throw std::domain_error(
        "Direction of integration does not match stepsize sign,"
        " adjust the direction or stepsize so that integration happens from xi to xf.");
  }
  // Check that yeval and x_eval are right size
  constexpr bool dense_output = compile_size_v<Vec> != 0;
  if constexpr (dense_output) {
    if (!x_eval.size()) {
      throw std::domain_error("Dense output requested but x_eval is size 0!");
    }
    // TODO: Better error messages
    auto x_eval_max = (direction * x_eval.maxCoeff());
    auto x_eval_min = (direction * x_eval.minCoeff());
    auto xi_intdir = direction * xi;
    auto xf_intdir = direction * xf;
    const bool high_range_err = xf_intdir < x_eval_max;
    const bool low_range_err = xi_intdir > x_eval_min;
    if (high_range_err || low_range_err) {
      if (high_range_err && low_range_err) {
        throw std::out_of_range(
            std::string{"The max and min of the output points ("}
            + std::to_string(x_eval_min) + std::string{", "}
            + std::to_string(x_eval_max)
            + ") lie outside the high and low of the integration range ("
            + std::to_string(xi_intdir) + std::string{", "}
            + std::to_string(xf_intdir) + ")!");
      }
      if (high_range_err) {
        throw std::out_of_range(
            std::string{"The max of the output points ("}
            + std::to_string(x_eval_max)
            + ") lies outside the high of the integration range ("
            + std::to_string(xf_intdir) + ")!");
      }
      if (low_range_err) {
        throw std::out_of_range(
            std::string{"The min of the output points ("}
            + std::to_string(x_eval_min)
            + ") lies outside the low of the integration range ("
            + std::to_string(xi_intdir) + ")!");
      }
    }
  }

  // Initialize vectors for storing results
  std::size_t output_size = 100;
  using stdvecd_t = std::vector<Scalar>;
  stdvecd_t xs;
  xs.reserve(output_size + 1);
  xs.push_back(xi);
  using complex_t = std::complex<Scalar>;
  using stdvecc_t = std::vector<complex_t>;
  stdvecc_t ys;
  ys.reserve(output_size);
  ys.push_back(yi);
  stdvecc_t dys;
  dys.reserve(output_size);
  dys.push_back(dyi);
  std::vector<int> successes;
  successes.reserve(output_size);
  successes.push_back(1);
  std::vector<int> steptypes;
  steptypes.reserve(output_size);
  steptypes.push_back(0);
  stdvecd_t phases;
  phases.reserve(output_size);
  using vectorc_t = vector_t<complex_t>;
  vectorc_t yeval(x_eval.size());
  vectorc_t dyeval(x_eval.size());

  complex_t y = yi;
  complex_t dy = dyi;
  complex_t yprev = y;
  complex_t dyprev = dy;
  auto scale_xi = scale(info.xp().array(), xi, init_stepsize).eval();
  auto omega_is = omega(info, scale_xi).eval();
  auto gamma_is = gamma(info, scale_xi).eval();
  Scalar wi = omega_is.mean();
  Scalar gi = gamma_is.mean();
  Scalar dwi = (2.0 / init_stepsize * (info.Dn() * omega_is)).mean();
  Scalar dgi = (2.0 / init_stepsize * (info.Dn() * gamma_is)).mean();
  Scalar hslo_ini = direction
                    * std::min(static_cast<Scalar>(1e8),
                               static_cast<Scalar>(std::abs(1.0 / wi)));
  Scalar hosc_ini
      = direction
        * std::min(std::min(static_cast<Scalar>(1e8),
                            static_cast<Scalar>(std::abs(wi / dwi))),
                   std::abs(gi / dgi));

  if (hard_stop) {
    if (direction * (xi + hosc_ini) > direction * xf) {
      hosc_ini = xf - xi;
    }
    if (direction * (xi + hslo_ini) > direction * xf) {
      hslo_ini = xf - xi;
    }
  }
  auto hslo = choose_nonosc_stepsize(info, xi, hslo_ini, Scalar(0.2));
  // o and g written here
  auto osc_step_tup = choose_osc_stepsize(info, xi, hosc_ini, epsilon_h);
  auto hosc = std::get<0>(osc_step_tup);
  // NOTE: Calling choose_osc_stepsize will update these values
  auto &&omega_n = std::get<1>(osc_step_tup);
  auto &&gamma_n = std::get<2>(osc_step_tup);
  Scalar xcurrent = xi;
  Scalar wnext = wi;
  using matrixc_t = matrix_t<complex_t>;
  matrixc_t y_eval;
  matrixc_t dy_eval;
  std::pair<complex_t, complex_t> a_pair;
  std::array<std::pair<LogInfo, std::size_t>, 5> solver_counts = info.info();
  while (std::abs(xcurrent - xf) > Scalar(1e-8)
         && direction * xcurrent < direction * xf) {
    Scalar phase{0.0};
    bool success = false;
    bool steptype = true;
    Scalar err;
    int cheb_N = 0;
    arena_matrix<vectorc_t> un(info.alloc_, omega_n.size(), 1);
    arena_matrix<vectorc_t> d_un(info.alloc_, omega_n.size(), 1);
    if (direction * hosc > direction * hslo){
      if (hard_stop) {
        if (direction * (xcurrent + hosc) > direction * xf) {
          hosc = xf - xcurrent;
          auto xp_scaled = scale(info.xp().array(), xcurrent, hosc).eval();
          omega_n = omega(info, xp_scaled);
          gamma_n = gamma(info, xp_scaled);
        }
        if (direction * (xcurrent + hslo) > direction * xf) {
          hslo = xf - xcurrent;
        }
      }
      // o and g read here
      std::tie(success, y, dy, err, phase, un, d_un, a_pair)
          = osc_step<dense_output>(info, omega_n, gamma_n, xcurrent, hosc,
                                   yprev, dyprev, eps);
      steptype = 1;
      solver_counts[get_idx(LogInfo::RICCSTEP)].second++;
    }
    if (unlikely(log_level == LogLevel::INFO && !success)) {
      info.logger().template log<LogLevel::INFO>(
        std::string("[Non-oscillatory step][x = ") +
        std::to_string(xcurrent) + std::string("][stepsize =") +
        std::to_string(hslo) + std::string("]")
      );
    } else if (unlikely(log_level == LogLevel::INFO)) {
      info.logger().template log<LogLevel::INFO>(
        std::string("[Oscillatory step][x = ") +
        std::to_string(xcurrent) + std::string("][stepsize =") +
        std::to_string(hslo) + std::string("]")
      );
    }
    while (!success) {
      std::tie(success, y, dy, err, y_eval, dy_eval, cheb_N)
          = nonosc_step(info, xcurrent, hslo, yprev, dyprev, eps);
      solver_counts[get_idx(LogInfo::CHEBSTEP)].second++;
      solver_counts[get_idx(LogInfo::LS)].second += cheb_N + 1;
      steptype = 0;
      if (!success) {
        hslo *= Scalar{0.5};
      }
      if (direction * hslo < std::numeric_limits<Scalar>::min()) {
        throw std::domain_error(std::string("Stepsize became to small error: ")
                                + std::to_string(direction * hslo));
      }
    }
    auto h = steptype ? hosc : hslo;
    if constexpr (dense_output) {
      Eigen::Index dense_size = 0;
      Eigen::Index dense_start = 0;
      // Assuming x_eval is sorted we just want start and size
      std::tie(dense_start, dense_size)
          = get_slice(x_eval, xcurrent, (xcurrent + h));
      if (dense_size != 0) {
        if (unlikely(log_level == LogLevel::INFO)) {
          info.logger().template log<LogLevel::INFO>(
            std::string("[Dense output][x_start = ") +
            std::to_string(xcurrent) + std::string("][x_end =") +
            std::to_string(xcurrent + h) + std::string("]")
          );
        }
        auto x_eval_map
            = Eigen::Map<vectord_t>(x_eval.data() + dense_start, dense_size);
        auto y_eval_map
            = Eigen::Map<vectorc_t>(yeval.data() + dense_start, dense_size);
        auto dy_eval_map
            = Eigen::Map<vectorc_t>(dyeval.data() + dense_start, dense_size);
        if (steptype) {
          auto x_eval_scaled = eval(
              info.alloc_,
              (2.0 / h * (x_eval_map.array() - xcurrent) - 1.0).matrix());
          auto Linterp = interpolate(info.xn(), x_eval_scaled, info.alloc_);
          auto fdense
              = eval(info.alloc_, (Linterp * un).array().exp().matrix());
          y_eval_map
              = a_pair.first * fdense + a_pair.second * fdense.conjugate();
          auto du_dense = eval(info.alloc_, (Linterp * d_un));
          dy_eval_map = a_pair.first * (du_dense.array() * fdense.array())
                        + a_pair.second
                              * (du_dense.array() * fdense.array()).conjugate();
        } else {
          auto xc_scaled
              = eval(info.alloc_,
                     scale(std::get<2>(info.chebyshev_[cheb_N]), xcurrent, h)
                         .matrix());
          auto Linterp = interpolate(xc_scaled, x_eval_map, info.alloc_);
          y_eval_map = Linterp * y_eval;
          dy_eval_map = Linterp * dy_eval;
        }
      }
    }
    // Finish appending and ending conditions
    ys.push_back(y);
    dys.push_back(dy);
    xs.push_back(xcurrent + h);
    phases.push_back(phase);
    steptypes.push_back(steptype);
    successes.push_back(success);
    Scalar dwnext;
    Scalar gnext;
    Scalar dgnext;
    if (steptype) {
      wnext = omega_n[0];
      gnext = gamma_n[0];
      dwnext = 2.0 / h * info.Dn().row(0).dot(omega_n);
      dgnext = 2.0 / h * info.Dn().row(0).dot(gamma_n);
    } else {
      wnext = omega(info, xcurrent + h);
      gnext = gamma(info, xcurrent + h);
      auto xn_scaled = scale(info.xn().array(), xcurrent, h).eval();
      dwnext = 2.0 / h * info.Dn().row(0).dot(omega(info, xn_scaled).matrix());
      dgnext
          = 2.0 / h * info.Dn().row(0).dot(gamma(info, (xn_scaled).matrix()));
    }
    xcurrent += h;
    if (direction * xcurrent < direction * xf) {
      hslo_ini
          = direction * std::min(Scalar{1e8}, std::abs(Scalar{1.0} / wnext));
      hosc_ini = direction
                 * std::min(std::min(Scalar{1e8}, std::abs(wnext / dwnext)),
                            std::abs(gnext / dgnext));
      if (hard_stop) {
        if (direction * (xcurrent + hosc_ini) > direction * xf) {
          hosc_ini = xf - xcurrent;
        }
        if (direction * (xcurrent + hslo_ini) > direction * xf) {
          hslo_ini = xf - xcurrent;
        }
      }
      // o and g written here
      osc_step_tup = choose_osc_stepsize(info, xcurrent, hosc_ini, epsilon_h);
      hosc = std::get<0>(osc_step_tup);
      hslo = choose_nonosc_stepsize(info, xcurrent, hslo_ini, Scalar{0.2});
      yprev = y;
      dyprev = dy;
    }
    info.alloc_.recover_memory();
  }
  if constexpr (!std::is_same_v<std::decay_t<decltype(info.logger())>, EmptyLogger>) {
    if (unlikely(log_level == LogLevel::INFO)) {
      std::size_t riccati_steps = 0;
      std::size_t rk_steps = 0;
      for (auto& success : successes) {
        if (success) {
          riccati_steps++;
        } else {
          rk_steps++;
        }
      }
      info.logger().template log<LogLevel::INFO>(
        std::string("Total Steps = ") + std::to_string(successes.size())
      );
      for (auto&& info_pair : solver_counts) {
        info.logger().template log<LogLevel::INFO>(
          std::string(to_string(info_pair.first)) + " = " + std::to_string(info_pair.second)
        );
      }
    }
  }
  return std::make_tuple(xs, ys, dys, successes, phases, steptypes, yeval,
                        dyeval, 1.0);
}


}  // namespace riccati

#endif
