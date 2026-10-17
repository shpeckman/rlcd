# src/rlcd/policy.cr
module Rlcd
  record Forward, hidden : Array(Float64), logits : Array(Float64), probs : Array(Float64)

  class Policy
    getter input_dim    : Int32
    getter hidden_dim   : Int32
    getter option_count : Int32
    getter w1           : Mat
    getter w2           : Mat
    getter b1           : Array(Float64)
    getter b2           : Array(Float64)

    def initialize(@input_dim : Int32, @hidden_dim : Int32, @option_count : Int32, @rng : Random = Random.new)
      @w1 = Mat.random(@hidden_dim, @input_dim, @rng)
      @b1 = Array(Float64).new(@hidden_dim, 0.0)
      @w2 = Mat.random(@option_count, @hidden_dim, @rng)
      @b2 = Array(Float64).new(@option_count, 0.0)
    end

    def forward(x : Array(Float64)) : Forward
      z1     = @w1.mul_vec(x)
      hidden = Array(Float64).new(@hidden_dim) { |i| Math.tanh(z1[i] + @b1[i]) }
      z2     = @w2.mul_vec(hidden)
      logits = Array(Float64).new(@option_count) { |i| z2[i] + @b2[i] }
      Forward.new(hidden, logits, softmax(logits))
    end

    def probs(x : Array(Float64)) : Array(Float64)
      forward(x).probs
    end

    def sample(x : Array(Float64)) : Int32
      categorical(forward(x).probs)
    end

    def softmax(logits : Array(Float64)) : Array(Float64)
      max  = logits.max
      exps = logits.map { |v| Math.exp(v - max) }
      sum  = exps.sum
      exps.map { |v| v / sum }
    end

    def categorical(probs : Array(Float64)) : Int32
      r   = @rng.next_float
      acc = 0.0
      probs.each_with_index do |p, i|
        acc += p
        return i if r < acc
      end
      probs.size - 1
    end
  end
end
