# src/rlcd/trainer.cr
module Rlcd
  record Example, features : Array(Float64), label : Int32

  record Stats, epoch : Int32, accuracy : Float64, mean_reward : Float64, mean_score : Float64

  class Trainer
    def initialize(
      @policy              : Policy,
      @lr                  : Float64     = 0.005,
      @group_size          : Int32       = 8,
      @scoring             : ScoringRule = ScoringRule::Brier,
      @calibration_weight  : Float64     = 1.0,
      @pg_weight           : Float64     = 0.0,
      @normalize_advantage : Bool        = true,
      @batch_size          : Int32       = 32,
      @lr_decay            : Float64     = 0.0,
      @rng                 : Random      = Random.new,
    )
      @adam = Adam.new(@policy, lr: @lr)
    end

    def train(examples : Array(Example), epochs : Int32, & : Stats ->) : Nil
      epochs.times do |epoch|
        apply_lr_decay(epoch + 1, epochs)
        yield train_epoch(examples, epoch + 1)
      end
    end

    def train(examples : Array(Example), epochs : Int32) : Nil
      train(examples, epochs) { }
    end

    def train_epoch(examples : Array(Example), epoch : Int32 = 0) : Stats
      shuffled    = examples.shuffle(@rng)
      correct     = 0
      reward_sum  = 0.0
      score_sum   = 0.0
      total       = 0
      batch_count = 0

      acc = @policy.param_slices.map { |s| Array(Float64).new(s.size, 0.0) }

      shuffled.each do |example|
        fwd   = @policy.forward(example.features)
        probs = fwd.probs
        label = example.label

        actions = Array(Int32).new(@group_size)
        rewards = Array(Float64).new(@group_size)
        @group_size.times do |i|
          a = @policy.categorical(probs)
          actions << a
          rewards << (a == label ? 1.0 : 0.0)
        end

        mean = rewards.sum / @group_size
        std  = Math.sqrt(rewards.sum { |r| (r - mean) ** 2 } / @group_size)
        advantages = rewards.map do |r|
          adv = r - mean
          (@normalize_advantage && std > 1e-8) ? adv / std : adv
        end

        n       = probs.size
        dlogits = Array(Float64).new(n, 0.0)
        if @pg_weight > 0.0
          @group_size.times do |i|
            adv = advantages[i] / @group_size
            a   = actions[i]
            n.times { |k| dlogits[k] -= @pg_weight * adv * ((k == a ? 1.0 : 0.0) - probs[k]) }
          end
        end

        sgrad = Rlcd.score_grad_logits(@scoring, probs, label)
        n.times { |k| dlogits[k] -= @calibration_weight * sgrad[k] }

        grads = backprop(example.features, fwd, dlogits)
        grads.each_with_index do |g, i|
          g.size.times { |j| acc[i][j] += g[j] }
        end
        batch_count += 1

        if batch_count >= @batch_size
          @adam.step!(acc.map { |a| Slice(Float64).new(a.to_unsafe, a.size) })
          acc.each { |a| a.fill(0.0) }
          batch_count = 0
        end

        rewards.each do |r|
          correct += 1 if r == 1.0
          reward_sum += r
          total += 1
        end
        score_sum += Rlcd.score(@scoring, probs, label)
      end

      if batch_count > 0
        @adam.step!(acc.map { |a| Slice(Float64).new(a.to_unsafe, a.size) })
      end

      Stats.new(
        epoch: epoch,
        accuracy: correct.to_f / total,
        mean_reward: reward_sum / total,
        mean_score: score_sum / shuffled.size,
      )
    end

    def backprop(x : Array(Float64), fwd : Forward, dlogits : Array(Float64)) : Array(Array(Float64))
      if @policy.hidden_dim == 0
        gw2 = Mat.new(@policy.option_count, @policy.input_dim)
        gw2.add_outer!(dlogits, x)
        [gw2.data.to_a, dlogits]
      else
        gw2 = Mat.new(@policy.option_count, @policy.hidden_dim)
        gw2.add_outer!(dlogits, fwd.hidden)
        dh  = @policy.w2.t_mul_vec(dlogits)
        dz1 = Array(Float64).new(@policy.hidden_dim) { |i| dh[i] * (1.0 - fwd.hidden[i] ** 2) }
        gw1 = Mat.new(@policy.hidden_dim, @policy.input_dim)
        gw1.add_outer!(dz1, x)
        [gw1.data.to_a, dz1, gw2.data.to_a, dlogits]
      end
    end

    private def apply_lr_decay(epoch : Int32, total : Int32) : Nil
      return if @lr_decay <= 0.0 || total <= 1
      frac = (epoch - 1).to_f / (total - 1).to_f
      @adam.lr = @lr * (1.0 - @lr_decay * frac)
    end
  end
end
