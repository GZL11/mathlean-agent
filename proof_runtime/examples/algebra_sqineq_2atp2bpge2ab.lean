import Mathlib

/-- For any two real numbers a and b, show that $a^2 + b^2 \geq 2ab$. -/
theorem algebra_sqineq_2atp2bpge2ab (a b : ℝ) : a ^ 2 + b ^ 2 ≥ 2 * a * b := by
  have h : 0 ≤ (a - b) ^ 2 := sq_nonneg (a - b)
  calc a ^ 2 + b ^ 2
      = a ^ 2 - 2 * a * b + b ^ 2 + 2 * a * b := by ring
    _ = (a - b) ^ 2 + 2 * a * b := by ring
    _ ≥ 0 + 2 * a * b := by linarith
    _ = 2 * a * b := by ring
