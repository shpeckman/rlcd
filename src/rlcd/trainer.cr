# src/rlcd/trainer.cr
module RLCD
  record Example, features : Array(Float64), label : Int32

  record Stats, epoch : Int32, accuracy : Float64, mean_reward : Float64, mean_score : Float64

  class Trainer
    getter optimizer : Adam
    getter epoch     : Int32 = 0
    getter seed      : UInt64?

    @lr      : Float64
    @epochs  : Int32 = 0
    @dlogits : Array(Float64)
    @sgrad   : Array(Float64)
    @dh      : Array(Float64)
    @dz1     : Array(Float64)
    @gw1     : Mat
    @gw2     : Mat

    def initialize(
      @policy              : Policy,
      lr                   : Float64?    = nil,
      @group_size          : Int32       = 8,
      @scoring             : ScoringRule = ScoringRule::Brier,
      @calibration_weight  : Float64     = 1.0,
      @pg_weight           : Float64     = 0.0,
      @normalize_advantage : Bool        = true,
      @batch_size          : Int32       = 32,
      @lr_decay            : Float64     = 0.0,
      @rng                 : Random      = Random.new,
      optimizer            : Adam?       = nil,
      @seed                : UInt64?     = nil,
    )
      if optimizer && !optimizer.policy.same?(@policy)
        raise ArgumentError.new("optimizer belongs to a different policy")
      end
      @lr           = lr || optimizer.try(&.lr) || 0.005
      @optimizer    = optimizer || Adam.new(@policy, lr: @lr)
      @optimizer.lr = @lr
      @dlogits = Array(Float64).new(@policy.option_count, 0.0)
      @sgrad   = Array(Float64).new(@policy.option_count, 0.0)
      if @policy.hidden_dim == 0
        @gw1 = Mat.new(0, 0)
        @gw2 = Mat.new(@policy.option_count, @policy.input_dim)
        @dh  = Array(Float64).new(0)
        @dz1 = Array(Float64).new(0)
      else
        @gw1 = Mat.new(@policy.hidden_dim, @policy.input_dim)
        @gw2 = Mat.new(@policy.option_count, @policy.hidden_dim)
        @dh  = Array(Float64).new(@policy.hidden_dim, 0.0)
        @dz1 = Array(Float64).new(@policy.hidden_dim, 0.0)
      end
    end

    def train(examples : Array(Example), epochs : Int32, start : Int32 = 0, & : Stats ->) : Nil
      raise ArgumentError.new("start #{start} is outside 0..#{epochs}") unless 0 <= start <= epochs
      @epoch  = start
      @epochs = epochs
      (start + 1).upto(epochs) do |epoch|
        apply_lr_decay(epoch, epochs)
        stats  = train_epoch(examples, epoch)
        @epoch = epoch
        yield stats
      end
    end

    def train(examples : Array(Example), epochs : Int32, start : Int32 = 0) : Nil
      train(examples, epochs, start) { }
    end

    def train(examples : Array(Example), checkpoint : Checkpoint, epochs : Int32 = checkpoint.epochs, & : Stats ->) : Nil
      raise ArgumentError.new("checkpoint seed #{checkpoint.seed.inspect} differs from trainer seed #{@seed.inspect}") unless checkpoint.seed == @seed
      train(examples, epochs, checkpoint.epoch) { |stats| yield stats }
    end

    def train(examples : Array(Example), checkpoint : Checkpoint, epochs : Int32 = checkpoint.epochs) : Nil
      train(examples, checkpoint, epochs) { }
    end

    def checkpoint : Checkpoint
      Checkpoint.new(@epoch, @epochs, @seed)
    end

    def train_epoch(examples : Array(Example), epoch : Int32 = 0) : Stats
      shuffled    = examples.shuffle(epoch_rng(epoch))
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
        n     = probs.size
        sc    = RLCD.score(@scoring, probs, label)

        @dlogits.fill(0.0)
        if @pg_weight > 0.0
          actions = Array(Int32).new(@group_size)
          rewards = Array(Float64).new(@group_size)
          @group_size.times do
            a = @policy.categorical(probs)
            actions << a
            rewards << (a == label ? 1.0 : 0.0)
          end

          mean = rewards.sum / @group_size
          std  = Math.sqrt(rewards.sum { |r| (r - mean) ** 2 } / @group_size)
          @group_size.times do |i|
            adv = rewards[i] - mean
            adv /= std if @normalize_advantage && std > 1e-8
            a = actions[i]
            n.times { |k| @dlogits[k] -= @pg_weight * (adv / @group_size) * ((k == a ? 1.0 : 0.0) - probs[k]) }
          end

          rewards.each do |r|
            correct += 1 if r == 1.0
            reward_sum += r
            total += 1
          end
        else
          correct += 1 if probs.index(probs.max) == label
          reward_sum += sc
          total += 1
        end

        RLCD.score_grad_logits(@scoring, probs, label, @sgrad)
        n.times { |k| @dlogits[k] -= @calibration_weight * @sgrad[k] }

        backprop(example.features, fwd, @dlogits, acc)
        batch_count += 1

        if batch_count >= @batch_size
          @optimizer.step!(acc.map { |a| Slice(Float64).new(a.to_unsafe, a.size) })
          acc.each { |a| a.fill(0.0) }
          batch_count = 0
        end

        score_sum += sc
      end

      if batch_count > 0
        @optimizer.step!(acc.map { |a| Slice(Float64).new(a.to_unsafe, a.size) })
      end

      Stats.new(
        epoch: epoch,
        accuracy: correct.to_f / total,
        mean_reward: reward_sum / total,
        mean_score: score_sum / shuffled.size,
      )
    end

    def backprop(x : Array(Float64), fwd : Forward, dlogits : Array(Float64), acc : Array(Array(Float64))) : Nil
      if @policy.hidden_dim == 0
        @gw2.data.fill(0.0)
        @gw2.add_outer!(dlogits, x)
        @gw2.data.each_with_index { |v, i| acc[0][i] += v }
        @policy.option_count.times { |i| acc[1][i] += dlogits[i] }
      else
        @gw2.data.fill(0.0)
        @gw2.add_outer!(dlogits, fwd.hidden)
        @gw2.data.each_with_index { |v, i| acc[2][i] += v }

        @policy.w2.t_mul_vec_into(dlogits, @dh)
        @policy.hidden_dim.times { |i| @dz1[i] = @dh[i] * (1.0 - fwd.hidden[i] ** 2) }

        @gw1.data.fill(0.0)
        @gw1.add_outer!(@dz1, x)
        @gw1.data.each_with_index { |v, i| acc[0][i] += v }

        @policy.hidden_dim.times { |i| acc[1][i] += @dz1[i] }
        @policy.option_count.times { |i| acc[3][i] += dlogits[i] }
      end
    end

    private def epoch_rng(epoch : Int32) : Random
      (seed = @seed) ? Random::PCG32.new(seed, epoch.to_u64) : @rng
    end

    private def apply_lr_decay(epoch : Int32, total : Int32) : Nil
      return if @lr_decay <= 0.0 || total <= 1
      frac = (epoch - 1).to_f / (total - 1).to_f
      @optimizer.lr = @lr * (1.0 - @lr_decay * frac)
    end
  end
end
