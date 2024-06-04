(** MA_CH **)

(* 读入库文件 *)
Require Export Real_Axioms.

Parameter R_stru : Real_struct.
Parameter R_axio : Real_axioms R_stru. 

Global Hint Resolve R_axio : core.

Declare Scope MA_R_scope.
Delimit Scope MA_R_scope with ma.
Open Scope MA_R_scope.

Definition ℝ := @R R_stru.
Definition ℕ := @N R_stru.
Definition ℤ := @Z R_stru.
Definition ℚ := @Q R_stru.
Definition fp := @fp R_stru.
Definition zeroR := @zeroR R_stru.
Definition fm := @fm R_stru.
Definition oneR := @oneR R_stru.
Definition Leq := @Leq R_stru.
Definition Abs := @Abs R_stru.

Notation "x + y" := fp[[x, y]] : MA_R_scope.
Notation "0" := zeroR : MA_R_scope.
Notation "x · y" := fm[[x, y]](at level 40) : MA_R_scope.
Notation "1" := oneR : MA_R_scope.
Notation "x ≤ y" := ([x, y] ∈ Leq)(at level 77) : MA_R_scope.
Notation "- a" := (∩(\{ λ u, u ∈ ℝ /\ u + a = 0 \})) : MA_R_scope.
Notation "x - y" := (x + (-y)) : MA_R_scope.
Notation "a ⁻" := (∩(\{ λ u, u ∈ (ℝ ~ [0]) /\ u · a = 1 \}))
  (at level 5) : MA_R_scope.
Notation "m / n" := (m · (n⁻)) : MA_R_scope.
Notation "｜ x ｜" := (Abs[x])(at level 5, x at level 0) : MA_R_scope.

Definition LT x y := x ≤ y /\ x <> y.
Notation "x < y" := (LT x y) : MA_R_scope.
Definition Ensemble_R := @Ensemble_R R_stru R_axio.
Definition PlusR := @PlusR R_stru R_axio.
Definition zero_in_R := @zero_in_R R_stru R_axio.
Definition Plus_P1 := @Plus_P1 R_stru R_axio.
Definition Plus_P2 := @Plus_P2 R_stru R_axio.
Definition Plus_P3 := @Plus_P3 R_stru R_axio.
Definition Plus_P4 := @Plus_P4 R_stru R_axio.
Definition MultR := @MultR R_stru R_axio.
Definition one_in_R := @one_in_R R_stru R_axio.
Definition Mult_P1 := @Mult_P1 R_stru R_axio.
Definition Mult_P2 := @Mult_P2 R_stru R_axio.
Definition Mult_P3 := @Mult_P3 R_stru R_axio.
Definition Mult_P4 := @Mult_P4 R_stru R_axio.
Definition Mult_P5 := @Mult_P5 R_stru R_axio.
Definition LeqR := @LeqR R_stru R_axio.
Definition Leq_P1 := @Leq_P1 R_stru R_axio.
Definition Leq_P2 := @Leq_P2 R_stru R_axio.
Definition Leq_P3 := @Leq_P3 R_stru R_axio.
Definition Leq_P4 := @Leq_P4 R_stru R_axio.
Definition Plus_Leq := @Plus_Leq R_stru R_axio.
Definition Mult_Leq := @Mult_Leq R_stru R_axio.
Definition Completeness := @Completeness R_stru R_axio.

Definition Plus_close := @Plus_close R_stru R_axio.
Definition Mult_close := @Mult_close R_stru R_axio.
Definition one_in_R_Co := @one_in_R_Co R_stru R_axio.
Definition Plus_Co1 := @Plus_Co1 R_stru R_axio.
Definition Plus_Co2 := @Plus_Co2 R_stru R_axio.
Definition Plus_neg1a := @Plus_neg1a R_stru R_axio.
Definition Plus_neg1b := @Plus_neg1b R_stru R_axio.
Definition Plus_neg2 := @Plus_neg2 R_stru R_axio.
Definition Minus_P1 := @Minus_P1 R_stru R_axio.
Definition Minus_P2 := @Minus_P2 R_stru R_axio.
Definition Plus_Co3 := @Plus_Co3 R_stru R_axio.
Definition Mult_Co1 := @Mult_Co1 R_stru R_axio.
Definition Mult_Co2 := @Mult_Co2 R_stru R_axio.
Definition Mult_inv1 := @Mult_inv1 R_stru R_axio.
Definition Mult_inv2 := @Mult_inv2 R_stru R_axio.
Definition Divide_P1 := @Divide_P1 R_stru R_axio.
Definition Divide_P2 := @Divide_P2 R_stru R_axio.
Definition Mult_Co3 := @Mult_Co3 R_stru R_axio.
Definition PlusMult_Co1 := @PlusMult_Co1 R_stru R_axio.
Definition PlusMult_Co2 := @PlusMult_Co2 R_stru R_axio.
Definition PlusMult_Co3 := @PlusMult_Co3 R_stru R_axio.
Definition PlusMult_Co4 := @PlusMult_Co4 R_stru R_axio.
Definition PlusMult_Co5 := @PlusMult_Co5 R_stru R_axio.
Definition PlusMult_Co6 := @PlusMult_Co6 R_stru R_axio.
Definition Order_Co1 := @Order_Co1 R_stru R_axio.
Definition Order_Co2 := @Order_Co2 R_stru R_axio.
Definition OrderPM_Co1 := @OrderPM_Co1 R_stru R_axio.
Definition OrderPM_Co2a := @OrderPM_Co2a R_stru R_axio.
Definition OrderPM_Co2b := @OrderPM_Co2b R_stru R_axio.
Definition OrderPM_Co3 := @OrderPM_Co3 R_stru R_axio.
Definition OrderPM_Co4 := @OrderPM_Co4 R_stru R_axio.
Definition OrderPM_Co5 := @OrderPM_Co5 R_stru R_axio.
Definition OrderPM_Co6 := @OrderPM_Co6 R_stru R_axio.
Definition OrderPM_Co7a := @OrderPM_Co7a R_stru R_axio.
Definition OrderPM_Co7b := @OrderPM_Co7b R_stru R_axio.
Definition OrderPM_Co8a := @OrderPM_Co8a R_stru R_axio.
Definition OrderPM_Co8b := @OrderPM_Co8b R_stru R_axio.
Definition OrderPM_Co9 := @OrderPM_Co9 R_stru R_axio.
Definition OrderPM_Co10 := @OrderPM_Co10 R_stru R_axio.
Definition OrderPM_Co11 := @OrderPM_Co11 R_stru R_axio.
Definition IndSet := @IndSet R_stru.
Definition IndSet_P1 := @IndSet_P1 R_stru.
Definition N_Subset_R := @N_Subset_R R_stru R_axio.
Definition one_in_N := @one_in_N R_stru R_axio.
Definition zero_not_in_N := @zero_not_in_N R_stru R_axio.
Definition IndSet_N := @IndSet_N R_stru R_axio.
Definition MathInd := @MathInd R_stru R_axio.
Definition Nat_P1a := @Nat_P1a R_stru R_axio.
Definition Nat_P2 := @Nat_P2 R_stru R_axio.
Definition Nat_P3 := @Nat_P3 R_stru R_axio.
Definition Nat_P4 := @Nat_P4 R_stru R_axio.
Definition Nat_P5 := @Nat_P5 R_stru R_axio.
Definition Nat_P6 := @Nat_P6 R_stru R_axio.
Definition one_is_min_in_N := @one_is_min_in_N R_stru R_axio.
Definition N_Subset_Z := @N_Subset_Z R_stru.
Definition Z_Subset_R := @Z_Subset_R R_stru R_axio.
Definition Int_P1_Lemma := @Int_P1_Lemma R_stru R_axio.
Definition Int_P1a := @Int_P1a R_stru R_axio.
Definition Int_P1b := @Int_P1b R_stru R_axio.
Definition Int_P2 := @Int_P2 R_stru R_axio.
Definition Int_P3 := @Int_P3 R_stru R_axio.
Definition Int_P4 := @Int_P4 R_stru R_axio.
Definition Int_P5 := @Int_P5 R_stru R_axio.
Definition Z_Subset_Q := @Z_Subset_Q R_stru R_axio.
Definition Q_Subset_R := @Q_Subset_R R_stru R_axio.
Definition Frac_P1 := @Frac_P1 R_stru R_axio.
Definition Frac_P2 := @Frac_P2 R_stru R_axio.
Definition Rat_P1a := @Rat_P1a R_stru R_axio.
Definition Rat_P1b := @Rat_P1b R_stru R_axio.
Definition Rat_P2 := @Rat_P2 R_stru R_axio.
Definition Rat_P3 := @Rat_P3 R_stru R_axio.
Definition Rat_P4 := @Rat_P4 R_stru R_axio.
Definition Rat_P5 := @Rat_P5 R_stru R_axio.
Definition Rat_P6 := @Rat_P6 R_stru R_axio.
Definition Rat_P7 := @Rat_P7 R_stru R_axio.
Definition Rat_P8 := @Rat_P8 R_stru R_axio.
Definition Rat_P9 := @Rat_P9 R_stru R_axio.
Definition Rat_P10 := @Rat_P10 R_stru R_axio.
Definition Even := @Even R_stru.
Definition Odd := @Odd R_stru.
Definition Even_and_Odd_P1 := @Even_and_Odd_P1 R_stru R_axio.
Definition Even_and_Odd_P2_Lemma := @Even_and_Odd_P2_Lemma R_stru R_axio.
Definition Even_and_Odd_P2 := @Even_and_Odd_P2 R_stru R_axio.
Definition Even_and_Odd_P3 := @Even_and_Odd_P3 R_stru R_axio.
Definition Existence_of_irRational_Number :=
  @Existence_of_irRational_Number R_stru R_axio.
Definition Arch_P1 := @Arch_P1 R_stru R_axio.
Definition Arch_P2 := @Arch_P2 R_stru R_axio.
Definition Arch_P3_Lemma := @Arch_P3_Lemma R_stru R_axio.
Definition Arch_P3a := @Arch_P3a R_stru R_axio.
Definition Arch_P3b := @Arch_P3b R_stru R_axio.
Definition Arch_P4 := @Arch_P4 R_stru R_axio.
Definition Arch_P5 := @Arch_P5 R_stru R_axio.
Definition Arch_P6 := @Arch_P6 R_stru R_axio.
Definition Arch_P7 := @Arch_P7 R_stru R_axio.
Definition Arch_P8 := @Arch_P8 R_stru R_axio.
Definition Arch_P9 := @Arch_P9 R_stru R_axio.
Definition Arch_P10 := @Arch_P10 R_stru R_axio.
Definition Abs_is_Function := @Abs_is_Function R_stru R_axio.
Definition Abs_in_R := @Abs_in_R R_stru R_axio.
Definition Distance := @Distance R_stru.
Definition me_zero_Abs := @me_zero_Abs R_stru R_axio.
Definition le_zero_Abs := @le_zero_Abs R_stru R_axio.
Definition Abs_P1 := @Abs_P1 R_stru R_axio.
Definition Abs_P2 := @Abs_P2 R_stru R_axio.
Definition Abs_P3 := @Abs_P3 R_stru R_axio.
Definition Abs_P4 := @Abs_P4 R_stru R_axio.
Definition Abs_P5 := @Abs_P5 R_stru R_axio.
Definition Abs_P6 := @Abs_P6 R_stru R_axio.
Definition Abs_P7 := @Abs_P7 R_stru R_axio.
Definition Abs_P8 := @Abs_P8 R_stru R_axio.

Global Hint Resolve Plus_close zero_in_R Mult_close one_in_R one_in_R_Co
  Plus_neg1a Plus_neg1b Plus_neg2 Minus_P1 Minus_P2
  Mult_inv1 Mult_inv2 Divide_P1 Divide_P2 OrderPM_Co9
  N_Subset_R one_in_N Nat_P1a Nat_P1b
  N_Subset_Z Z_Subset_R Int_P1a Int_P1b
  Z_Subset_Q Q_Subset_R Rat_P1a Rat_P1b Abs_in_R: real.
  
(* 1.2 数集 确界原理 *)

(* 1.2.1 区间与邻域 *)

(* 有限区间 *) 
(* 开区间 *)
Notation "］ a , b ［" := (\{ λ x, x ∈ ℝ /\ a < x /\ x < b \})
  (at level 5, a at level 0, b at level 0) : MA_R_scope.

(* 闭区间 *)
Notation "［ a , b ］" := (\{ λ x, x ∈ ℝ /\ a ≤ x /\ x ≤ b \})
  (at level 5, a at level 0, b at level 0) : MA_R_scope.

(* 左开右闭 *)
Notation "］ a , b ］" := (\{ λ x, x ∈ ℝ /\ a < x /\ x ≤ b \})
  (at level 5, a at level 0, b at level 0) : MA_R_scope.

(* 左闭右开 *)
Notation "［ a , b ［" := (\{ λ x, x ∈ ℝ /\ a ≤ x /\ x < b \})
  (at level 5, a at level 0, b at level 0) : MA_R_scope.

(* 无限区间 *)
Notation "］ a , +∞［" := (\{ λ x, x ∈ ℝ /\ a < x \})
  (at level 5, a at level 0) : MA_R_scope.

Notation "［ a , +∞［" := (\{ λ x, x ∈ ℝ /\ a ≤ x \})
  (at level 5, a at level 0) : MA_R_scope.

Notation "］-∞ , b ］" := (\{ λ x, x ∈ ℝ /\ x ≤ b \})
  (at level 5, b at level 0) : MA_R_scope.

Notation "］-∞ , b ［" := (\{ λ x, x ∈ ℝ /\ x < b \})
  (at level 5, b at level 0) : MA_R_scope.

Notation "]-∞ , +∞[" := ℝ (at level 0) : MA_R_scope.

(* 邻域 *)
Definition Neighbourhood x δ := x ∈ ℝ /\ δ ∈ ℝ /\ x ∈ ］(x - δ),(x + δ)［.

(* 邻域 *)
Definition Neighbor a δ := \{ λ x, x ∈ ℝ /\ ｜ (x - a) ｜ < δ \}.
Notation "U( a ; δ )" := (Neighbor a δ)
  (a at level 0, δ at level 0, at level 4) : MA_R_scope.

(* 左邻域 *)
Definition leftNeighbor a δ := ］a-δ, a］.

(* 右邻域 *)
Definition rightNeighbor a δ := ［a, (a+δ)［.

(* 去心邻域 *)
Definition Neighbor0 a δ := \{ λ x, x ∈ ℝ 
  /\ 0 < ｜(x-a)｜ /\ ｜(x-a)｜ < δ \}.
Notation "Uº( a ; δ )" := (Neighbor0 a δ)
  (a at level 0, δ at level 0, at level 4) : MA_R_scope.

(* 左去心邻域 *)
Definition leftNeighbor0 a δ := ］a-δ, a［.
Notation "U-º( a ; δ )" := (leftNeighbor0 a δ)
  (a at level 0, δ at level 0, at level 4) : MA_R_scope.

(* 右去心邻域 *)
Definition rightNeighbor0 a δ := ］a, (a+δ)［.
Notation "U+º( a ; δ )" := (rightNeighbor0 a δ)
  (a at level 0, δ at level 0, at level 4) : MA_R_scope.

(* 无穷邻域 *)
Definition Neighbor_infinity M := \{ λ x, x ∈ ℝ /\ M ∈ ℝ
  /\ 0 < M /\ M < ｜x｜ \}.
Notation "U(∞) M" := (Neighbor_infinity M) (at level 5) : MA_R_scope.

(* 正无穷邻域 *)
Definition PNeighbor_infinity M := \{ λ x, x ∈ ℝ /\ M ∈ ℝ /\ M < x \}.
Notation "U(+∞) M" := (［ M , +∞［) (at level 5) : MA_R_scope.

(* 负无穷邻域 *)
Definition NNeighbor_infinity M := \{ λ x, x ∈ ℝ /\ M ∈ ℝ /\ x < M \}.
Notation "U(-∞) M" := (］-∞ , M ［) (at level 5) : MA_R_scope.

(* 1.2.2 有界集 确界原理 *)

(* 上界 *)
Definition UpperBound S M := S ⊂ ℝ /\ M ∈ ℝ /\ (∀ x, x ∈ S -> x ≤ M).

(* 下界 *)
Definition LowerBound S L := S ⊂ ℝ /\ L ∈ ℝ /\ (∀ x, x ∈ S -> L ≤ x).

(* 有界集 *)
Definition Bounded S := ∃ M L, UpperBound S M /\ LowerBound S L.

(* 无界集 *)
Definition Unbounded S := ~ (Bounded S).

(* 上确界 *)
Definition Sup S η := UpperBound S η /\ (∀ α, α ∈ ℝ -> α < η
  -> (∃ x0, x0 ∈ S /\ α < x0)).

(* 下确界 *)
Definition Inf S ξ := LowerBound S ξ /\ (∀ β, β ∈ ℝ -> ξ < β
  -> (∃ x0, x0 ∈ S /\ x0 < β)).
  
Definition Max S c := S ⊂ ℝ /\ c ∈ S /\ (∀ x, x ∈ S -> x ≤ c).

Definition Min S c := S ⊂ ℝ /\ c ∈ S /\ (∀ x, x ∈ S -> c ≤ x).

Corollary Max_Corollary : ∀ S c1 c2, Max S c1 -> Max S c2 -> c1 = c2.
Proof.
  intros. unfold Max in *. destruct H as [H []], H0 as [H0 []].
  pose proof H1. pose proof H3. apply H2 in H6. apply H4 in H5.
  apply Leq_P2; auto.
Qed.
  
Corollary Min_Corollary : ∀ S c1 c2, Min S c1 -> Min S c2 -> c1 = c2.
Proof.
  intros. unfold Min in *. destruct H as [H []], H0 as [H0 []].
  pose proof H1. pose proof H3. apply H2 in H6. apply H4 in H5.
  apply Leq_P2; auto.
Qed.

Definition Sup_Eq S η := Min \{ λ u, UpperBound S u \} η.

Corollary Sup_Corollary : ∀ S η, Sup S η <-> Sup_Eq S η.
Proof.
  intros. split; intro.
  - red in H; red. destruct H, H as [H []]. repeat split. 
    unfold Included; intros. apply AxiomII in H3 as [_[_[]]]; auto.
    apply AxiomII; split. unfold Ensemble; exists ℝ; auto. split; auto.
    intros. apply AxiomII in H3 as [H3 [H4 []]]. pose proof H5.
    apply (Order_Co1 x η) in H7 as [H7 | [|]]; auto.
    + apply H0 in H7 as [x0 []]; auto. pose proof H7. apply H6 in H9.
      destruct H8. elim H10. apply Leq_P2; auto.
    + destruct H7. auto.
    + rewrite H7. apply Leq_P1; auto.
  - red in H; red. destruct H as [H []]. apply AxiomII in H0 as [_[H0 []]].
    repeat split; auto. intros. apply NNPP; intro.
    assert (∀ x1, x1 ∈ S -> x1 ≤ α).
    { intros. apply NNPP; intro. elim H6. exists x1. split; auto.  
      pose proof H4. apply (@ Order_Co1 x1 α) in H9
      as [H9 | [|]]; auto. elim H8. destruct H9. auto. 
      rewrite H9 in H8. elim H8. apply Leq_P1; auto. }
    assert (α ∈ \{ λ u, UpperBound S u \}).
    { apply AxiomII. split. exists ℝ; auto. split; auto. }
    apply H1 in H8. destruct H5. elim H9. 
    apply Leq_P2; auto.
Qed.

Definition Inf_Eq S ξ := Max \{ λ u, LowerBound S u \} ξ.

Corollary Inf_Corollary : ∀ S ξ, Inf S ξ <-> Inf_Eq S ξ.
Proof.
  intros. split; intro.
  - red in H; red. destruct H, H as [H []]. repeat split.
    unfold Included; intros. apply AxiomII in H3 as [_[_[]]]; auto.
    apply AxiomII; split. exists ℝ; auto. repeat split; auto.
    intros. apply AxiomII in H3 as [H3 [H4 []]]. pose proof H5.
    apply (@ Order_Co1 x ξ) in H7 as [H7 | [|]]; auto.
    + destruct H7. auto.
    + apply H0 in H7 as [x0 []]; auto. pose proof H7. apply H6 in H9.
      destruct H8. elim H10. apply Leq_P2; auto.
    + rewrite H7. apply Leq_P1; auto.
  - red in H; red. destruct H as [H []]. apply AxiomII in H0 as [_[H2 []]].
    repeat split; auto. intros. apply NNPP; intro.
    assert (∀x1, x1 ∈ S -> β ≤ x1).
    { intros. apply NNPP; intro. elim H6. exists x1. split; auto.
      pose proof H4. apply (Order_Co1 x1 β) in H9 
      as [H9 | [|]]; auto. elim H8. destruct H9. auto.
      rewrite H9 in H8. elim H8. apply Leq_P1; auto. }
    assert (β ∈ \{ λ u, LowerBound S u \}).
    { apply AxiomII; split. exists ℝ; auto. repeat split; auto. }
    apply H1 in H8. destruct H5. elim H9.
    apply Leq_P2; auto.
Qed.

(* 确界原理 *)

(* 上确界引理 *)
Lemma SupLemma : ∀ X, X ⊂ ℝ -> X <> Φ -> (∃ c, UpperBound X c) 
  -> exists ! η, Sup X η.
Proof.
  intros. set (Y:=\{ λ u, UpperBound X u \}).
  assert (Y <> Φ).
  { apply NEexE. destruct H1 as [x]. exists x. apply AxiomII;
    split; auto. destruct H1 as [_[]]. unfold Ensemble. exists ℝ; auto. }
  assert (Y ⊂ ℝ).
  { unfold Included; intros. apply AxiomII in H3 as [_].
    destruct H3 as [_[]]. auto. }
  assert (∃ c, c ∈ ℝ /\ (∀ x y, x ∈ X -> y ∈ Y 
    -> (x ≤ c /\ c ≤ y))) as [c[]].
  { apply Completeness; auto. intros. apply AxiomII in H5 as [_].
    destruct H5 as [_[]]. apply H6 in H4; auto. }
  assert (c ∈ Y).
  { apply AxiomII; repeat split; eauto. intros.
    apply NEexE in H2 as [y]. pose proof H5 _ _ H6 H2; tauto. }
  assert (Min \{ λ u, UpperBound X u \} c).
  { red. repeat split; auto. intros y H7. apply NEexE in H0 as [x].
    pose proof H5 _ _ H0 H7. tauto. }
  exists c. split. apply Sup_Corollary. auto. intros.
  apply Sup_Corollary in H8. apply (Min_Corollary Y); auto.
Qed.

(* 下确界引理 *)
Lemma InfLemma : ∀ X, X ⊂ ℝ -> X <> Φ -> (∃ c, LowerBound X c)
  -> exists ! ξ, Inf X ξ.
Proof.
  intros. set(Y:=\{ λ u, LowerBound X u \}).
  assert (Y <> Φ).
  { apply NEexE. destruct H1 as [x]. exists x. apply AxiomII;
    split; auto. destruct H1 as [_[]]. exists ℝ; auto. }
  assert (Y ⊂ ℝ).
  { unfold Included. intros. apply AxiomII in H3 as [].
    destruct H4 as [_[]]; auto. }
  assert (∃ c, c ∈ ℝ /\ (∀ y x, y ∈ Y -> x ∈ X 
    -> y ≤ c /\ c ≤ x)) as [c[]].
  { apply Completeness; auto. intros.
    apply AxiomII in H4 as [_[_[]]]. apply H6 in H5; auto. }
  assert (c ∈ Y).
  { apply AxiomII. repeat split; eauto. intros. 
    apply NEexE in H2 as [y]. pose proof H5 _ _ H2 H6; tauto. }
  assert (Max \{ λ u, LowerBound X u \} c).
  { red. repeat split; auto. intros y H7. apply NEexE in H0 as [x].
    pose proof H5 _ _ H7 H0; tauto. }
  exists c. split. apply Inf_Corollary; auto. intros.
  apply Inf_Corollary in H8. apply (Max_Corollary Y); auto.
Qed.

(* 确界原理 *)
Theorem Sup_Inf_Principle : ∀ X, X ⊂ ℝ -> X <> Φ
  -> ((∃ c, UpperBound X c) -> exists ! η, Sup X η)
  /\ ((∃ c, LowerBound X c) -> exists ! ξ, Inf X ξ).
Proof.
  intros. split; intros.
  - apply SupLemma; auto.
  - apply InfLemma; auto.
Qed.

(* 1.3 函数概念 *)

(* 1.3.1 函数的定义 *)
(* Note ：MK中已经给出 *)

(* 有序数对 *)
Definition Ordered x y := [ [x] | [x|y] ].
Notation "[ x , y ]" := (Ordered x y) (at level 0) : MA_R_scope.

(* 以有序数对的第一个元为第一坐标 *)
Definition First z := ∩∩z.

(* 以有序数对的第二个元为第二坐标 *)
Definition Second z := (∩∪z)∪(∪∪z) ~ (∪∩z).

(* 有序数对相等，对应坐标相等 *)
Theorem ProdEqual : ∀ x y u v, Ensemble x -> Ensemble y
  -> ([x, y] = [u, v] <-> x = u /\ y = v).
Proof.
  apply MKT55.
Qed.

(* 关系 *)
Definition Relation r := ∀ z, z ∈ r -> (∃ x y, z = [x, y]).

(* 关系的复合及关系的逆 *)
Definition Composition r s := \{\ λ x z, ∃ y, [x,y] ∈ s /\ [y,z] ∈ r \}\.
Notation "r ∘ s" := (Composition r s) (at level 50).

Definition Inverse r := \{\ λ x y, [y, x] ∈ r \}\.
Notation "r ⁻¹" := (Inverse r) (at level 5).

(* 满足性质P的有序数对构成的集合: { (x,y) : ... } *)
Notation "\{\ P \}\" :=
  (\{ λ z, ∃ x y, z = [x,y] /\ P x y \}) (at level 0).

(* 分类公理图示II关于有序数对的适应性事实 *)
Fact AxiomII' : ∀ a b P,
  [a,b] ∈ \{\ P \}\ <-> Ensemble ([a,b]) /\ (P a b).
Proof.
  apply AxiomII'.
Qed.

(* 函数 *)
Definition Function f := 
  Relation f /\ (∀ x y z, [x,y] ∈ f -> [x,z] ∈ f -> y = z).

(* 定义域 *)
Definition Domain f := \{ λ x, ∃ y, [x,y] ∈ f \}.
Notation "dom( f )" := (Domain f) (at level 5).

(* 值域 *)
Definition Range f := \{ λ y, ∃ x, [x,y] ∈ f \}.
Notation "ran( f )" := (Range f) (at level 5).

(* f在点x的函数值 *)
Definition Value f x := ∩\{ λ y, [x,y] ∈ f \}.
Notation "f [ x ]" := (Value f x) (at level 5).

(* 1.3.2 函数的四则运算 *)

Definition Plus_Fun f g :=
  \{\ λ x y, x ∈ dom(f) /\ x ∈ dom(g) /\ y = f[x] + g[x] \}\.
Definition Sub_Fun f g :=
  \{\ λ x y, x ∈ dom(f) /\ x ∈ dom(g) /\ y = f[x] - g[x] \}\.
Definition Mult_Fun f g :=
  \{\ λ x y, x ∈ dom(f) /\ x ∈ dom(g) /\ y = f[x] · g[x] \}\.
Definition Div_Fun f g :=
  \{\ λ x y, x ∈ dom(f) /\ x ∈ dom(g) /\ g[x] <> 0 /\ y = f[x] / g[x] \}\.
  
Notation "f \+ g" := (Plus_Fun f g) (at level 45, left associativity).
Notation "f \- g" := (Sub_Fun f g) (at level 45, left associativity).
Notation "f \· g" := (Mult_Fun f g) (at level 40, left associativity).
Notation "f // g" := (Div_Fun f g) (at level 40, left associativity).

(* 1.3.3 复合函数 *)

Definition Comp : ∀ f g, Function f -> Function g -> Function (f ∘ g).
Proof.
  apply MKT64.
Qed.

(* 1.3.4 反函数 *)

Definition Inverse_Fun f g := Function1_1 f /\ g = f⁻¹.

Corollary Inverse_Co1 : ∀ f u, Function f -> Function f⁻¹ -> u ∈ dom(f)
  -> (f⁻¹)[f[u]] = u.
Proof.
  intros. apply Property_Value,invp1,Property_Fun in H1; auto.
Qed.

Corollary Inverse_Co2: ∀ f u, Function f -> Function f⁻¹ -> u ∈ ran(f)
  -> f[(f⁻¹)[u]] = u.
Proof.
  intros. rewrite reqdi in H1. apply Property_Value in H1; auto.
  apply ->invp1 in H1; auto. apply Property_Fun in H1; auto.
Qed.

(* 1.4 具有某些特性的函数 *)

(* 1.4.1 有界函数 *)

Definition UpBoundedFun f D : Prop :=
  Function f /\ D = dom(f) /\ (∃ M, M ∈ ℝ /\ ∀ x, x ∈ D -> f[x] ≤ M).

Definition LowBoundedFun f D : Prop :=
  Function f /\ D = dom(f) /\ (∃ L, L ∈ ℝ /\ ∀ x, x ∈ D -> L ≤ f[x]).

Definition BoundedFun f D : Prop := Function f /\ D = dom(f) 
  /\ (∃ M, M ∈ ℝ -> 0 < M /\ ∀ x, x ∈ D -> ｜[f[x]]｜ ≤ M).

(* 1.4.2 单调函数 *)

Definition IncreaseFun f := Function f
  /\ (∀ x1 x2, x1 ∈ dom(f) -> x2 ∈ dom(f) -> x1 < x2 -> f[x1] ≤ f[x2]).
Definition StrictIncreaseFun f := Function f
  /\ (∀ x1 x2, x1 ∈ dom(f) -> x2 ∈ dom(f) -> x1 < x2 -> f[x1] < f[x2]).
Definition DecreaseFun f := Function f
/\ (∀ x1 x2, x1 ∈ dom(f) -> x2 ∈ dom(f) -> x1 < x2 -> f[x2] ≤ f[x1]).
Definition StrictDecreaseFun f := Function f
  /\ (∀ x1 x2, x1 ∈ dom(f) -> x2 ∈ dom(f) -> x1 < x2 -> f[x2] < f[x1]).

Theorem Theorem1_2_1 : ∀ f, Function f -> dom(f) ⊂ ℝ -> ran(f) ⊂ ℝ
  -> StrictIncreaseFun f -> StrictIncreaseFun f⁻¹.
Proof.
  intros; unfold StrictIncreaseFun in *. destruct H2 as [H2 H3].
  assert (Function f⁻¹).
  { unfold Function in *. unfold Relation in *. destruct H as []; split.
    - intros. apply AxiomII in H5 as [H5 [x[y]]]. exists x, y; tauto.
    - intros. apply AxiomII' in H5,H6. destruct H5,H6.
      assert (y ∈ ℝ).
      { unfold Included in H0. apply H0. apply Property_dom in H7; auto. }
      assert (z ∈ ℝ).
      { apply H0. apply Property_dom in H8; auto. }
      New H7. New H8. apply Property_dom in H11, H12.
      destruct (Order_Co1 y z) as [H13 | [|]]; auto.
      + apply H3 in H13; auto.
        assert (x = f[y] /\ x = f[z]) as [].
        { split; [apply (H4 y)|apply (H4 z)]; auto;
          apply Property_Value; auto. }
        rewrite <-H14, <-H15 in H13. destruct H13. elim H16. auto.
      + apply H3 in H13; auto.
        assert (x = f[y] /\ x = f[z]) as [].
        { split; [apply (H4 y)|apply (H4 z)]; auto;
          apply Property_Value; auto. }
        rewrite <-H14, <-H15 in H13. destruct H13. elim H16. auto. }
  split; auto; intros.
  - New H5; New H6. apply Property_Value, Property_ran in H8, H9; auto.
    rewrite <-deqri in H8, H9.
    destruct (Order_Co1 (f⁻¹)[x1] (f⁻¹)[x2])
    as [H10 | [|]]; auto.
    + apply H3 in H10; auto. rewrite f11vi in H10, H10; auto;
      try rewrite reqdi; auto. destruct H7, H10. elim H12.
      rewrite <-reqdi in H5, H6. apply Leq_P2; auto.
    + assert (f[(f ⁻¹)[x1]] = f[(f ⁻¹)[x2]]). { rewrite H10. auto. }
      rewrite <-reqdi in H5, H6. rewrite f11vi in H11, H11; auto.
      destruct H7. elim H12. auto.
Qed.

Theorem Theorem1_2_2 : ∀ f, Function f -> dom(f) ⊂ ℝ -> ran(f) ⊂ ℝ 
  -> StrictDecreaseFun f -> StrictDecreaseFun f⁻¹.
Proof.
  intros. unfold StrictDecreaseFun in *. destruct H2 as [].
  assert (Function f⁻¹).
  { unfold Function in *. unfold Relation in *. destruct H as []; split.
    + intros. apply AxiomII in H5 as [H5 [x[y]]]. exists x, y; tauto.
    + intros. apply AxiomII' in H5 as [], H6 as [].
      New H7; New H8. apply Property_dom in H9, H10.
      destruct (Order_Co1 y z) as [H11 | [|]]; auto.
      * apply H3 in H11; auto.
        assert (x = f[z] /\ x = f[y]) as [].
        { split; [apply (H4 z)|apply (H4 y)]; auto;
          apply Property_Value; auto. }
        rewrite <-H12, <-H13 in H11. destruct H11. elim H14. auto.
      * apply H3 in H11; auto.
        assert (x = f[z] /\ x = f[y]) as [].
        { split; [apply (H4 z)|apply (H4 y)]; auto;
          apply Property_Value; auto. }
        rewrite <-H12, <-H13 in H11. destruct H11. elim H14. auto. }
  split; auto; intros.
  - New H5; New H6. apply Property_Value,Property_ran in H8, H9; auto.
    rewrite <-deqri in H8, H9.
    destruct (Order_Co1 (f⁻¹)[x2] (f⁻¹)[x1]) 
    as [H10 | [|]]; auto.
    + apply H3 in H10; auto. rewrite f11vi in H10, H10; auto;
      rewrite <-reqdi in H5, H6; auto. destruct H7, H10. elim H11.
      apply Leq_P2; auto.
    + assert (f[(f ⁻¹)[x2]] = f[(f ⁻¹)[x1]]). { rewrite H10. auto. }
      rewrite <-reqdi in H5, H6. rewrite f11vi in H11, H11; auto.
      destruct H7. elim H12; auto.
Qed.

(* 1.4.3 奇函数和偶函数 *)

(* 奇函数 *)
Definition OddFun f := Function f /\ dom(f) ⊂ ℝ /\ ran(f) ⊂ ℝ
  /\ (∀ x, x ∈ dom(f) -> f[-x] = -f[x]).

(* 偶函数 *)
Definition EvenFun f := Function f /\ dom(f) ⊂ ℝ /\ ran(f) ⊂ ℝ
  /\ (∀ x, x ∈ dom(f) -> f[-x] = f[x]).

(* 1.4.4 周期函数 *)
Definition PeriodicFun f := Function f /\ (∃ σ, σ ∈ ℝ -> 0 < σ 
  /\ (∀ x, x ∈ ℝ -> x ∈ dom(f) -> (x + σ ∈ dom(f) -> f[x + σ] = f[x])
  /\ (x - σ ∈ dom(f) -> f[x - σ] = f[x]))).















