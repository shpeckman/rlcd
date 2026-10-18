# src/rlcd/answer.cr
module RLCD
  record NoulAnswer, id : String, noul : Float64, probabilities : Array(Float64)
  record ChoiceAnswer, id : String, choice : Int32, probabilities : Array(Float64), confidence : Float64
  record ScoreAnswer, id : String, score : Float64, probabilities : Array(Float64), confidence : Float64, legend : Array(String)

  def self.confidence(probs : Array(Float64)) : Float64
    n = probs.size
    return 1.0 if n <= 1
    p_max = probs.max
    ((p_max - 1.0 / n) / (1.0 - 1.0 / n)).clamp(0.0, 1.0)
  end

  def self.expected_score(probs : Array(Float64)) : Float64
    s = 0.0
    probs.each_with_index { |p, i| s += p * i }
    s
  end

  def self.score_confidence(probs : Array(Float64)) : Float64
    n = probs.size
    return 1.0 if n <= 1
    mean = expected_score(probs)
    mad  = 0.0
    probs.each_with_index { |p, i| mad += (i - mean).abs * p }
    uniform_mean = (n - 1) / 2.0
    uniform_mad  = 0.0
    (0...n).each { |i| uniform_mad += (i - uniform_mean).abs * (1.0 / n) }
    (1.0 - mad / uniform_mad).clamp(0.0, 1.0)
  end
end
