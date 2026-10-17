# src/rlcd/scoring.cr
module Rlcd
  enum ScoringRule
    LogScore
    Brier
  end

  def self.score(rule : ScoringRule, probs : Array(Float64), label : Int32) : Float64
    case rule
    in .log_score?
      Math.log(probs[label].clamp(1e-12, 1.0))
    in .brier?
      sum = 0.0
      probs.each_with_index do |p, i|
        d = p - (i == label ? 1.0 : 0.0)
        sum += d * d
      end
      -sum
    end
  end

  def self.score_grad_logits(rule : ScoringRule, probs : Array(Float64), label : Int32) : Array(Float64)
    n = probs.size
    case rule
    in .log_score?
      Array(Float64).new(n) { |k| (k == label ? 1.0 : 0.0) - probs[k] }
    in .brier?
      dot = 0.0
      n.times { |j| dot += probs[j] * (probs[j] - (j == label ? 1.0 : 0.0)) }
      Array(Float64).new(n) do |k|
        yk = k == label ? 1.0 : 0.0
        -2.0 * probs[k] * ((probs[k] - yk) - dot)
      end
    end
  end
end
