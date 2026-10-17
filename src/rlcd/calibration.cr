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

  def self.temperature_probs(logits : Array(Float64), t : Float64) : Array(Float64)
    scaled = logits.map { |z| z / t }
    max    = scaled.max
    exps   = scaled.map { |z| Math.exp(z - max) }
    sum    = exps.sum
    exps.map { |e| e / sum }
  end

  def self.fit_temperature(logits : Array(Array(Float64)), labels : Array(Int32)) : Float64
    return 1.0 if logits.empty?

    nll = ->(s : Float64) do
      total = 0.0
      logits.each_with_index do |z, i|
        zs    = z.map { |v| v * s }
        m     = zs.max
        exps  = zs.map { |v| Math.exp(v - m) }
        denom = exps.sum
        total += Math.log(exps[labels[i]] / denom)
      end
      -total / logits.size
    end

    phi = (1.0 + Math.sqrt(5.0)) / 2.0
    a   = 0.05
    b   = 20.0
    c   = b - (b - a) / phi
    d   = a + (b - a) / phi
    60.times do
      if nll.call(c) < nll.call(d)
        b = d
      else
        a = c
      end
      c = b - (b - a) / phi
      d = a + (b - a) / phi
    end
    1.0 / ((a + b) / 2.0)
  end

  def self.write_bins_csv(bins : Array(Bin), path : String, ece : Float64) : Nil
    File.open(path, "w") do |f|
      f.puts "# ece=#{ece}"
      f.puts "lower,upper,count,avg_confidence,accuracy,gap"
      bins.each do |bin|
        next if bin.count == 0
        f.puts "#{bin.lower},#{bin.upper},#{bin.count},#{bin.avg_confidence},#{bin.accuracy},#{bin.gap}"
      end
    end
  end
end
