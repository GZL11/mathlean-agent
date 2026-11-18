/-- 证明逻辑中的恒等律：如果 P 为真，则 P 为真 -/
theorem basic_logic (P : Prop) : P → P := by
  intro h
  exact h