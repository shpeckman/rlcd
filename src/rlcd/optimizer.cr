# src/rlcd/optimizer.cr
module Rlcd
  class Adam
    @t      : Int32
    @params : Array(Slice(Float64))
    @m      : Array(Slice(Float64))
    @v      : Array(Slice(Float64))

    def initialize(
      policy : Policy,
      @lr    : Float64 = 0.005,
      @beta1 : Float64 = 0.9,
      @beta2 : Float64 = 0.999,
      @eps   : Float64 = 1e-8,
    )
      @t = 0
      @params = [
        policy.w1.data,
        Slice(Float64).new(policy.b1.to_unsafe, policy.b1.size),
        policy.w2.data,
        Slice(Float64).new(policy.b2.to_unsafe, policy.b2.size),
      ]
      @m = @params.map { |p| Slice(Float64).new(p.size, 0.0) }
      @v = @params.map { |p| Slice(Float64).new(p.size, 0.0) }
    end

    def step!(grads : Array(Slice(Float64))) : Nil
      @t += 1
      bc1 = 1.0 - @beta1 ** @t
      bc2 = 1.0 - @beta2 ** @t
      @params.each_with_index do |p, i|
        g  = grads[i]
        mi = @m[i]
        vi = @v[i]
        p.size.times do |j|
          gj = g[j]
          mi[j] = @beta1 * mi[j] + (1.0 - @beta1) * gj
          vi[j] = @beta2 * vi[j] + (1.0 - @beta2) * gj * gj
          p[j] -= @lr * (mi[j] / bc1) / (Math.sqrt(vi[j] / bc2) + @eps)
        end
      end
    end
  end
end
