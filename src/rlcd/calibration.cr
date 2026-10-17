# src/rlcd/calibration.cr
module Rlcd
  alias Prediction = Tuple(Array(Float64), Int32)

  record Bin, lower : Float64, upper : Float64, count : Int32, avg_confidence : Float64, accuracy : Float64 do
    def gap : Float64
      (avg_confidence - accuracy).abs
    end
  end

  def self.ece(predictions : Array(Prediction), bins : Int32 = 10) : {Float64, Array(Bin)}
    conf_sum = Array(Float64).new(bins, 0.0)
    hit_sum  = Array(Float64).new(bins, 0.0)
    counts   = Array(Int32).new(bins, 0)

    predictions.each do |(probs, label)|
      best = 0
      (1...probs.size).each { |i| best = i if probs[i] > probs[best] }
      conf = probs[best]
      idx  = {(conf * bins).to_i, bins - 1}.min
      counts[idx] += 1
      conf_sum[idx] += conf
      hit_sum[idx] += 1.0 if best == label
    end

    result = Array(Bin).new(bins)
    ece    = 0.0
    total  = predictions.size
    bins.times do |i|
      count    = counts[i]
      avg_conf = count > 0 ? conf_sum[i] / count : 0.0
      acc      = count > 0 ? hit_sum[i] / count : 0.0
      bin = Bin.new(
        lower: i.to_f / bins,
        upper: (i + 1).to_f / bins,
        count: count,
        avg_confidence: avg_conf,
        accuracy: acc,
      )
      result << bin
      ece += (count.to_f / total) * bin.gap if count > 0
    end

    {ece, result}
  end
end
