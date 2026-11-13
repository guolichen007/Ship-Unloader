"""Coverage constraints keep their dimension through normal sign changes."""

import math
import unittest
import numpy as np
from scipy.optimize import minimize
from ship_perception.perception.constrained_rectangle import fixed_side_guard


class FixedSideGuardContracts(unittest.TestCase):
    def evaluate(self, x, points=None, axis=0, sign=-1):
        if points is None:
            points = np.array([[0.0, -4.0], [0.0, 1.0], [0.0, 1.0], [0.0, 7.0]])
        return fixed_side_guard(x, points, np.eye(2)[axis], axis, sign, 20.0, 0.2, 4.0)

    def test_dimension_and_finite_jacobian_across_den_zero(self):
        for angle in (
            0.0,
            math.pi / 2 - 1e-9,
            math.pi / 2,
            math.pi / 2 + 1e-9,
            math.pi,
        ):
            for axis in (0, 1):
                for sign in (-1, 1):
                    values, jac = self.evaluate(
                        np.array([angle, 0.1, -0.1, 0.0, -1.0]), axis=axis, sign=sign
                    )
                    self.assertEqual(values.shape, (9,))
                    self.assertEqual(jac.shape, (9, 5))
                    self.assertTrue(np.isfinite(jac).all())
                    self.assertTrue(np.isfinite(values).all())

    def test_analytic_derivative_at_perpendicular_and_opposite(self):
        for angle in (
            0.3,
            math.pi / 2 - 1e-5,
            math.pi / 2,
            math.pi / 2 + 1e-5,
            math.pi,
        ):
            x = np.array([angle, 0.1, -0.1, 0.0, -1.0])
            for axis in (0, 1):
                for sign in (-1, 1):
                    _, actual = self.evaluate(x, axis=axis, sign=sign)
                    numerical = []
                    for j in range(5):
                        step = np.eye(5)[j] * 1e-6
                        numerical.append(
                            (
                                self.evaluate(x + step, axis=axis, sign=sign)[0]
                                - self.evaluate(x - step, axis=axis, sign=sign)[0]
                            )
                            / 2e-6
                        )
                    np.testing.assert_allclose(
                        actual, np.array(numerical).T, atol=1e-7, rtol=1e-6
                    )

    def test_positive_den_preserves_original_coverage_sign(self):
        points = np.array([[0.0, -4.0], [0.0, 7.0]])
        for angle in (-1.0, 0.0, 1.0):
            x = np.array([angle, 0.1, -0.1, 0.0, -1.0])
            values, _ = self.evaluate(x, points=points)
            den = math.cos(angle)
            displacement = (
                20 * x[1] - 10 - points @ np.array([math.cos(angle), math.sin(angle)])
            ) / den
            np.testing.assert_allclose(values[1:3] / den, 0.2 - displacement)
            np.testing.assert_allclose(values[3:] / den, 4.0 + displacement)

    def test_optimizer_can_cross_initial_negative_den_without_dimension_error(self):
        shapes = set()

        def constraints(x):
            values, jac = self.evaluate(x)
            shapes.add((values.shape, jac.shape))
            return values

        result = minimize(
            lambda x: (x[0] - 0.1) ** 2,
            np.array([1.7, -0.5, 0.0, 0.0, -1.0]),
            method="SLSQP",
            bounds=[(-2.0, 2.0), (None, None), (0.0, 0.0), (0.0, 0.0), (-1.0, -1.0)],
            constraints=[
                dict(type="ineq", fun=constraints, jac=lambda x: self.evaluate(x)[1])
            ],
        )
        self.assertEqual(shapes, {((9,), (9, 5))})
        self.assertTrue(np.isfinite(result.x).all())


if __name__ == "__main__":
    unittest.main()
